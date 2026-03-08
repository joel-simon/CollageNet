from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
LATENT_DOWNSAMPLE_FACTOR = 8


def collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    if input_dir.is_file():
        if input_dir.suffix.lower() not in IMAGE_EXTENSIONS:
            raise FileNotFoundError(f"Unsupported image file: {input_dir}")
        return [input_dir]
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def snap_to_multiple_of_8(value: int) -> int:
    if value < 8:
        raise ValueError("Image dimensions must be at least 8 pixels.")
    return max(8, value - (value % 8))


def resolve_target_size(
    original_width: int,
    original_height: int,
    max_width: int | None,
    max_height: int | None,
) -> tuple[int, int]:
    scale = 1.0

    if max_width is not None and original_width > max_width:
        scale = min(scale, max_width / original_width)
    if max_height is not None and original_height > max_height:
        scale = min(scale, max_height / original_height)

    scaled_width = max(8, math.floor(original_width * scale))
    scaled_height = max(8, math.floor(original_height * scale))

    target_width = snap_to_multiple_of_8(scaled_width)
    target_height = snap_to_multiple_of_8(scaled_height)
    return target_width, target_height


def load_image(
    image_path: Path,
    max_width: int | None,
    max_height: int | None,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_size = image.size
        target_size = resolve_target_size(
            original_width=original_size[0],
            original_height=original_size[1],
            max_width=max_width,
            max_height=max_height,
        )
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

    image_np = np.asarray(image).astype(np.float32) / 255.0
    image_np = image_np * 2.0 - 1.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    return image_tensor, original_size, target_size


def encode_image(
    vae: AutoencoderKL,
    image_tensor: torch.Tensor,
    device: str,
    mode: str,
) -> torch.Tensor:
    image_tensor = image_tensor.to(device=device, dtype=vae.dtype)
    with torch.inference_mode():
        posterior = vae.encode(image_tensor).latent_dist
        latents = posterior.mean if mode == "mean" else posterior.sample()
        latents = latents * vae.config.scaling_factor
    return latents.squeeze(0).detach().cpu().to(torch.float16)


def save_latent(
    output_path: Path,
    latent: torch.Tensor,
    source_path: Path,
    original_size: tuple[int, int],
    processed_size: tuple[int, int],
    mode: str,
    scaling_factor: float,
    vae_id: str = DEFAULT_VAE_ID,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_source_path = source_path.resolve()
    scale_x = original_size[0] / processed_size[0]
    scale_y = original_size[1] / processed_size[1]
    np.savez_compressed(
        output_path,
        latent=latent.numpy(),
        source_path=str(resolved_source_path),
        original_width=original_size[0],
        original_height=original_size[1],
        processed_width=processed_size[0],
        processed_height=processed_size[1],
        processed_to_original_scale_x=float(scale_x),
        processed_to_original_scale_y=float(scale_y),
        latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
        vae_id=vae_id,
        scaling_factor=float(scaling_factor),
        mode=mode,
    )


def prepare_latents_from_images(
    input_dir: Path,
    output_dir: Path,
    vae: AutoencoderKL,
    device: str,
    *,
    max_width: int | None = None,
    max_height: int | None = None,
    mode: str = "mean",
    recursive: bool = False,
    skip_existing: bool = False,
    vae_id: str = DEFAULT_VAE_ID,
) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input image source does not exist: {input_dir}")

    image_paths = collect_images(input_dir, recursive=recursive)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_root = input_dir if input_dir.is_dir() else input_dir.parent
    latent_paths: list[Path] = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(input_root)
        output_path = output_dir / relative_path.with_suffix(".npz")
        latent_paths.append(output_path)

        if skip_existing and output_path.exists():
            continue

        image_tensor, original_size, processed_size = load_image(
            image_path=image_path,
            max_width=max_width,
            max_height=max_height,
        )
        latent = encode_image(
            vae=vae,
            image_tensor=image_tensor,
            device=device,
            mode=mode,
        )
        save_latent(
            output_path=output_path,
            latent=latent,
            source_path=image_path,
            original_size=original_size,
            processed_size=processed_size,
            mode=mode,
            scaling_factor=vae.config.scaling_factor,
            vae_id=vae_id,
        )

    return latent_paths
