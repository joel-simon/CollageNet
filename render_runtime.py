from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from patch_dictionary_core import (
    DiffusionPatchProjector,
    LatentFelzenszwalbRegionProjector,
    LatentThresholdRegionProjector,
    build_patch_bank,
    render_pixel_collage_from_assignments,
    render_pixel_collage_from_region_assignments,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
PATCH_BANK_DTYPE = (
    torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device=DEVICE if DEVICE == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def load_pipeline(model_cfg: Any) -> StableDiffusionXLPipeline:
    vae = AutoencoderKL.from_pretrained(
        model_cfg.vae_id,
        torch_dtype=PIPELINE_DTYPE,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_cfg.model_id,
        vae=vae,
        torch_dtype=PIPELINE_DTYPE,
        use_safetensors=bool(getattr(model_cfg, "use_safetensors", True)),
    )

    if getattr(model_cfg, "lora_repo", None):
        pipe.load_lora_weights(model_cfg.lora_repo, weight_name=model_cfg.lora_weight_name)

        if getattr(model_cfg, "embedding_filename", None):
            embedding_path = hf_hub_download(
                repo_id=model_cfg.lora_repo,
                filename=model_cfg.embedding_filename,
                repo_type="model",
            )
            embedding_state = load_file(embedding_path)
            pipe.load_textual_inversion(
                embedding_state["clip_l"],
                token=model_cfg.embedding_token,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
            )
            pipe.load_textual_inversion(
                embedding_state["clip_g"],
                token=model_cfg.embedding_token,
                text_encoder=pipe.text_encoder_2,
                tokenizer=pipe.tokenizer_2,
            )

        pipe.fuse_lora()

    return pipe.to(DEVICE)


def make_projector_for_cfg(patch_cfg: Any, patch_bank: Any):
    mode = patch_cfg.projector_mode if hasattr(patch_cfg, "projector_mode") else patch_cfg.region_method
    if mode == "latent_threshold_regions":
        return LatentThresholdRegionProjector(cfg=patch_cfg, patch_bank=patch_bank)
    if mode == "latent_felzenszwalb":
        return LatentFelzenszwalbRegionProjector(cfg=patch_cfg, patch_bank=patch_bank)
    return DiffusionPatchProjector(cfg=patch_cfg, patch_bank=patch_bank)


def build_runtime_patch_bank(patch_cfg: Any):
    return build_patch_bank(
        patch_cfg,
        device=DEVICE,
        patch_bank_dtype=PATCH_BANK_DTYPE,
    )


def render_one(
    pipe: StableDiffusionXLPipeline,
    gen_cfg: Any,
    patch_cfg: Any,
    patch_bank: Any,
    output_dir: Path,
    *,
    final_filename: str | None = None,
    save_auxiliary_outputs: bool = False,
    save_displayed_image: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    projector = make_projector_for_cfg(patch_cfg, patch_bank)
    if hasattr(projector, "set_debug_output_dir"):
        debug_dir = output_dir / "debug_regions" if save_auxiliary_outputs else None
        projector.set_debug_output_dir(debug_dir)

    result = pipe(
        prompt=gen_cfg.prompt,
        negative_prompt=gen_cfg.negative_prompt,
        height=gen_cfg.height,
        width=gen_cfg.width,
        num_inference_steps=gen_cfg.num_inference_steps,
        guidance_scale=gen_cfg.guidance_scale,
        generator=make_generator(gen_cfg.seed),
        callback_on_step_end=projector,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    final_image = result.images[0]
    metadata = {
        "generation": asdict(gen_cfg),
        "projection": asdict(patch_cfg),
        "patch_bank_shape": list(patch_bank.raw_patches.shape),
        "projection_events": projector.projection_events,
    }

    sharp_render = None
    if projector.final_region_assignments is not None and projector.final_assignment_grid_shape is not None:
        sharp_render = render_pixel_collage_from_region_assignments(
            patch_bank=patch_bank,
            region_assignments=projector.final_region_assignments,
            latent_canvas_shape=projector.final_assignment_grid_shape,
            pixel_render_scale=gen_cfg.pixel_render_scale,
        )
        if save_auxiliary_outputs:
            region_path = output_dir / f"{gen_cfg.output_stem}_region_assignments.npy"
            np.save(region_path, np.array(projector.final_region_assignments, dtype=object), allow_pickle=True)
            sharp_path = output_dir / f"{gen_cfg.output_stem}_pixel_collage.png"
            sharp_render.save(sharp_path)
            metadata["region_assignments_path"] = str(region_path)
            metadata["pixel_collage_path"] = str(sharp_path)
            metadata["assignment_grid_shape"] = list(projector.final_assignment_grid_shape)
    elif projector.final_selected_patch_indices is not None and projector.final_assignment_grid_shape is not None:
        selected_patch_indices = projector.final_selected_patch_indices.numpy()
        sharp_render = render_pixel_collage_from_assignments(
            patch_bank=patch_bank,
            selected_patch_indices=selected_patch_indices[0],
            grid_shape=projector.final_assignment_grid_shape,
            patch_size=patch_cfg.patch_size,
            pixel_render_scale=gen_cfg.pixel_render_scale,
        )
        if save_auxiliary_outputs:
            selected_path = output_dir / f"{gen_cfg.output_stem}_selected_patch_indices.npy"
            np.save(selected_path, selected_patch_indices)
            sharp_path = output_dir / f"{gen_cfg.output_stem}_pixel_collage.png"
            sharp_render.save(sharp_path)
            metadata["selected_patch_indices_path"] = str(selected_path)
            metadata["pixel_collage_path"] = str(sharp_path)
            metadata["assignment_grid_shape"] = list(projector.final_assignment_grid_shape)

    output_image = sharp_render if (save_displayed_image and sharp_render is not None) else final_image
    final_path = output_dir / (final_filename or f"{gen_cfg.output_stem}.png")
    output_image.save(final_path)

    if save_auxiliary_outputs:
        save_json(metadata, output_dir / f"{gen_cfg.output_stem}.json")

    return {
        "final_path": str(final_path),
        "num_projection_events": len(projector.projection_events),
        "projector": type(projector).__name__,
    }
