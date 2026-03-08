from __future__ import annotations

import argparse
import hashlib
import re
import secrets
import tempfile
from dataclasses import replace
from pathlib import Path

from render_config import RenderConfig, load_render_config

CLI_OVERRIDE_FIELDS = {
    "prompt": ("generation", "prompt"),
    "negative_prompt": ("generation", "negative_prompt"),
    "source_images": ("source", "source_images"),
    "source_latents": ("source", "source_latents"),
    "recursive": ("source", "recursive"),
    "output_dir": ("output", "output_dir"),
    "output_stem": ("generation", "output_stem"),
    "seed": ("generation", "seed"),
    "num_seeds": ("generation", "num_seeds"),
    "seed_offset": ("generation", "seed_offset"),
    "height": ("generation", "height"),
    "width": ("generation", "width"),
    "num_inference_steps": ("generation", "num_inference_steps"),
    "guidance_scale": ("generation", "guidance_scale"),
    "region_method": ("projection", "region_method"),
    "projection_start_frac": ("projection", "projection_start_frac"),
    "projection_end_frac": ("projection", "projection_end_frac"),
    "projection_every_n_steps": ("projection", "projection_every_n_steps"),
    "alpha_start": ("projection", "alpha_start"),
    "alpha_end": ("projection", "alpha_end"),
    "dictionary_chunk_size": ("projection", "dictionary_chunk_size"),
    "similarity_temperature": ("projection", "similarity_temperature"),
    "preview_every_n_projections": ("projection", "preview_every_n_projections"),
    "random_seed": ("projection", "random_seed"),
    "felzenszwalb_scale": ("projection", "felzenszwalb_scale"),
    "felzenszwalb_sigma": ("projection", "felzenszwalb_sigma"),
    "felzenszwalb_min_size": ("projection", "felzenszwalb_min_size"),
    "region_candidate_count": ("projection", "region_candidate_count"),
    "region_min_area": ("projection", "region_min_area"),
    "region_max_area": ("projection", "region_max_area"),
    "region_max_bbox_h": ("projection", "region_max_bbox_h"),
    "region_max_bbox_w": ("projection", "region_max_bbox_w"),
    "debug_every_n_projections": ("projection", "debug_every_n_projections"),
    "threshold_min_regions": ("projection", "threshold_min_regions"),
    "threshold_max_regions": ("projection", "threshold_max_regions"),
    "threshold_connectivity": ("projection", "threshold_connectivity"),
    "threshold_similarity_low": ("projection", "threshold_similarity_low"),
    "threshold_similarity_high": ("projection", "threshold_similarity_high"),
    "total_patches": ("projection", "total_patches"),
    "patch_size": ("projection", "patch_size"),
    "max_source_image_width": ("source", "max_width"),
    "max_source_image_height": ("source", "max_height"),
    "do_rotated": ("projection", "do_rotated"),
    "save_auxiliary_outputs": ("output", "save_auxiliary_outputs"),
    "save_displayed_image": ("output", "save_displayed_image"),
}


def add_cli_override_argument(parser: argparse.ArgumentParser, name: str, **kwargs) -> None:
    options = [f"--{name.replace('_', '-')}"]
    underscored = f"--{name}"
    if underscored not in options:
        options.append(underscored)
    parser.add_argument(*options, dest=name, **kwargs)


def add_cli_boolean_argument(parser: argparse.ArgumentParser, name: str) -> None:
    options = [f"--{name.replace('_', '-')}"]
    underscored = f"--{name}"
    if underscored not in options:
        options.append(underscored)
    parser.add_argument(*options, action=argparse.BooleanOptionalAction, default=None, dest=name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render SDXL patch-collage images from YAML + CLI overrides.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config file.")
    add_cli_override_argument(parser, "prompt", type=str, default=None)
    add_cli_override_argument(parser, "negative_prompt", type=str, default=None)
    add_cli_override_argument(parser, "source_images", type=Path, default=None)
    add_cli_override_argument(parser, "source_latents", type=Path, default=None)
    add_cli_boolean_argument(parser, "recursive")
    add_cli_override_argument(parser, "output_dir", type=Path, default=None)
    add_cli_override_argument(parser, "output_stem", type=str, default=None)
    add_cli_override_argument(parser, "seed", type=int, default=None)
    add_cli_override_argument(parser, "num_seeds", type=int, default=None)
    add_cli_override_argument(parser, "seed_offset", type=int, default=None)
    add_cli_override_argument(parser, "height", type=int, default=None)
    add_cli_override_argument(parser, "width", type=int, default=None)
    add_cli_override_argument(parser, "num_inference_steps", type=int, default=None)
    add_cli_override_argument(parser, "guidance_scale", type=float, default=None)
    add_cli_override_argument(parser, "region_method", type=str, default=None)
    add_cli_override_argument(parser, "projection_start_frac", type=float, default=None)
    add_cli_override_argument(parser, "projection_end_frac", type=float, default=None)
    add_cli_override_argument(parser, "projection_every_n_steps", type=int, default=None)
    add_cli_override_argument(parser, "alpha_start", type=float, default=None)
    add_cli_override_argument(parser, "alpha_end", type=float, default=None)
    add_cli_override_argument(parser, "dictionary_chunk_size", type=int, default=None)
    add_cli_override_argument(parser, "similarity_temperature", type=float, default=None)
    add_cli_override_argument(parser, "preview_every_n_projections", type=int, default=None)
    add_cli_override_argument(parser, "random_seed", type=int, default=None)
    add_cli_override_argument(parser, "felzenszwalb_scale", type=float, default=None)
    add_cli_override_argument(parser, "felzenszwalb_sigma", type=float, default=None)
    add_cli_override_argument(parser, "felzenszwalb_min_size", type=int, default=None)
    add_cli_override_argument(parser, "region_candidate_count", type=int, default=None)
    add_cli_override_argument(parser, "region_min_area", type=int, default=None)
    add_cli_override_argument(parser, "region_max_area", type=int, default=None)
    add_cli_override_argument(parser, "region_max_bbox_h", type=int, default=None)
    add_cli_override_argument(parser, "region_max_bbox_w", type=int, default=None)
    add_cli_override_argument(parser, "debug_every_n_projections", type=int, default=None)
    add_cli_override_argument(parser, "threshold_min_regions", type=int, default=None)
    add_cli_override_argument(parser, "threshold_max_regions", type=int, default=None)
    add_cli_override_argument(parser, "threshold_connectivity", type=int, default=None)
    add_cli_override_argument(parser, "threshold_similarity_low", type=float, default=None)
    add_cli_override_argument(parser, "threshold_similarity_high", type=float, default=None)
    add_cli_override_argument(parser, "total_patches", type=int, default=None)
    add_cli_override_argument(parser, "patch_size", type=int, default=None)
    add_cli_override_argument(parser, "max_source_image_width", type=int, default=None)
    add_cli_override_argument(parser, "max_source_image_height", type=int, default=None)
    add_cli_boolean_argument(parser, "do_rotated")
    add_cli_boolean_argument(parser, "save_auxiliary_outputs")
    add_cli_boolean_argument(parser, "save_displayed_image")
    return parser.parse_args()


def _set_nested_attr(obj: object, path: tuple[str, ...], value: object) -> None:
    target = obj
    for name in path[:-1]:
        target = getattr(target, name)
    setattr(target, path[-1], value)


def apply_cli_overrides(cfg: RenderConfig, args: argparse.Namespace) -> RenderConfig:
    for arg_name, path in CLI_OVERRIDE_FIELDS.items():
        value = getattr(args, arg_name)
        if value is not None:
            _set_nested_attr(cfg, path, value)
    return cfg


def format_prompt_for_filename(prompt: str, max_length: int = 50) -> str:
    normalized = prompt.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        normalized = "render"
    return normalized[:max_length].rstrip("_") or "render"


def format_override_value_for_filename(value: object) -> str:
    if isinstance(value, Path):
        value = value.stem or value.name
    elif isinstance(value, bool):
        value = "true" if value else "false"
    elif isinstance(value, float):
        value = f"{value:g}"
    else:
        value = str(value)

    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "value"


def build_override_stub(args: argparse.Namespace, max_length: int = 120) -> str:
    skip_in_filename = {"prompt", "output_dir", "output_stem"}
    parts: list[str] = []

    for arg_name in CLI_OVERRIDE_FIELDS:
        if arg_name in skip_in_filename:
            continue
        value = getattr(args, arg_name, None)
        if value is None:
            continue
        key = format_override_value_for_filename(arg_name)
        val = format_override_value_for_filename(value)
        parts.append(f"{key}_{val}")

    if not parts:
        return ""

    full_stub = "__".join(parts)
    if len(full_stub) <= max_length:
        return full_stub

    digest = hashlib.sha1(full_stub.encode("utf-8")).hexdigest()[:8]
    truncated = full_stub[:max_length].rstrip("_")
    return f"{truncated}__h_{digest}"


def default_output_filename(prompt: str, seed: int, args: argparse.Namespace) -> str:
    prompt_stub = format_prompt_for_filename(prompt)
    override_stub = build_override_stub(args)
    random_id = secrets.token_hex(4)
    if override_stub:
        return f"{prompt_stub}__{override_stub}_seed_{seed:04d}_{random_id}.png"
    return f"{prompt_stub}_seed_{seed:04d}_{random_id}.png"


def validate_config(cfg: RenderConfig) -> None:
    source_latents = cfg.source.source_latents or cfg.projection.latent_dir
    source_images = cfg.source.source_images
    if bool(source_latents) == bool(source_images):
        raise ValueError("Set exactly one of source.source_images or source.source_latents.")
    if cfg.generation.num_seeds < 1:
        raise ValueError("generation.num_seeds must be >= 1")
    if cfg.generation.width % 8 != 0 or cfg.generation.height % 8 != 0:
        raise ValueError("generation.width and generation.height must both be divisible by 8")
    if cfg.generation.pixel_render_scale < 1:
        raise ValueError("generation.pixel_render_scale must be >= 1")
    if cfg.projection.threshold_connectivity not in {4, 8}:
        raise ValueError("projection.threshold_connectivity must be 4 or 8")


def resolve_latent_source(cfg: RenderConfig, pipe) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    from render_runtime import DEVICE
    from source_latents import prepare_latents_from_images

    source_latents = cfg.source.source_latents or cfg.projection.latent_dir
    if source_latents is not None:
        return Path(source_latents), None

    temp_dir = tempfile.TemporaryDirectory(prefix="sdxl_patch_collage_latents_")
    latent_dir = Path(temp_dir.name)
    prepare_latents_from_images(
        input_dir=Path(cfg.source.source_images),
        output_dir=latent_dir,
        vae=pipe.vae,
        device=DEVICE,
        max_width=cfg.source.max_width,
        max_height=cfg.source.max_height,
        mode=cfg.source.encode_mode,
        recursive=cfg.source.recursive,
        vae_id=cfg.model.vae_id,
    )
    return latent_dir, temp_dir


def main() -> None:
    args = parse_args()
    from render_runtime import (
        DEVICE,
        PATCH_BANK_DTYPE,
        PIPELINE_DTYPE,
        build_runtime_patch_bank,
        load_pipeline,
        render_one,
        set_seed,
    )

    cfg = apply_cli_overrides(load_render_config(args.config), args)
    validate_config(cfg)

    print(f"DEVICE={DEVICE}")
    print(f"PIPELINE_DTYPE={PIPELINE_DTYPE}")
    print(f"PATCH_BANK_DTYPE={PATCH_BANK_DTYPE}")

    set_seed(cfg.generation.seed)
    pipe = load_pipeline(cfg.model)

    latent_dir, temp_dir = resolve_latent_source(cfg, pipe)
    try:
        patch_cfg = replace(cfg.projection, latent_dir=latent_dir)
        patch_bank = build_runtime_patch_bank(patch_cfg)

        output_dir = Path(cfg.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        run_results = []
        start_seed = cfg.generation.seed + cfg.generation.seed_offset
        end_seed = start_seed + cfg.generation.num_seeds
        for seed in range(start_seed, end_seed):
            run_gen_cfg = replace(cfg.generation, seed=seed)
            if cfg.generation.output_stem:
                filename = (
                    f"{cfg.generation.output_stem}_seed_{seed:04d}.png"
                    if cfg.generation.num_seeds > 1
                    else f"{cfg.generation.output_stem}.png"
                )
            else:
                filename = default_output_filename(
                    prompt=run_gen_cfg.prompt,
                    seed=seed,
                    args=args,
                )
            print(f"rendering seed={seed} -> {filename}")
            run_result = render_one(
                pipe=pipe,
                gen_cfg=run_gen_cfg,
                patch_cfg=patch_cfg,
                patch_bank=patch_bank,
                output_dir=output_dir,
                final_filename=filename,
                save_auxiliary_outputs=cfg.output.save_auxiliary_outputs,
                save_displayed_image=cfg.output.save_displayed_image,
            )
            run_results.append(run_result)

        print(f"Saved {len(run_results)} image(s) to {output_dir}")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
