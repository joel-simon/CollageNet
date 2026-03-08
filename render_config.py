from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_id: str = "madebyollin/sdxl-vae-fp16-fix"
    lora_repo: str | None = None
    lora_weight_name: str | None = None
    embedding_filename: str | None = None
    embedding_token: list[str] = field(default_factory=list)
    use_safetensors: bool = True


@dataclass
class SourceConfig:
    source_images: Path | None = None
    source_latents: Path | None = None
    recursive: bool = False
    max_width: int | None = None
    max_height: int | None = None
    encode_mode: str = "mean"


@dataclass
class GenerationConfig:
    prompt: str = "a portrait on a white background."
    negative_prompt: str = "blurry, low quality, deformed, extra limbs, text, watermark"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 6.5
    seed: int = 0
    num_seeds: int = 1
    seed_offset: int = 0
    output_stem: str | None = None
    pixel_render_scale: int = 1


@dataclass
class ProjectionConfig:
    latent_dir: Path | None = None
    region_method: str = "felzenszwalb"
    patch_size: int = 1
    do_rotated: bool = False
    total_patches: int = 50000
    top_k: int = 1
    projection_start_frac: float = 0.7
    projection_end_frac: float = 1.0
    projection_every_n_steps: int = 1
    alpha_start: float = 0.0
    alpha_end: float = 0.1
    dictionary_chunk_size: int = 8192
    similarity_temperature: float = 0.10
    preview_every_n_projections: int = 0
    random_seed: int = 1234
    region_candidate_count: int = 128
    region_min_area: int = 1
    region_max_area: int = 1200
    region_max_bbox_h: int = 0
    region_max_bbox_w: int = 0
    debug_every_n_projections: int = 1
    threshold_min_regions: int = 64
    threshold_max_regions: int = 768
    threshold_connectivity: int = 4
    threshold_similarity_low: float = -0.25
    threshold_similarity_high: float = 0.999
    felzenszwalb_scale: float = 32.0
    felzenszwalb_sigma: float = 0.8
    felzenszwalb_min_size: int = 8

    @property
    def projector_mode(self) -> str:
        mapping = {
            "square": "square",
            "threshold": "latent_threshold_regions",
            "latent_threshold_regions": "latent_threshold_regions",
            "felzenszwalb": "latent_felzenszwalb",
            "latent_felzenszwalb": "latent_felzenszwalb",
        }
        try:
            return mapping[self.region_method]
        except KeyError as exc:
            raise ValueError(f"Unsupported region_method: {self.region_method}") from exc


@dataclass
class OutputConfig:
    output_dir: Path = Path("outputs/render")
    save_auxiliary_outputs: bool = False
    save_displayed_image: bool = True


@dataclass
class RenderConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _maybe_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def _update_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    for key, value in updates.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown config key: {type(instance).__name__}.{key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def _resolve_path(path: Path | None, base_dir: Path) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (base_dir / path).resolve()


def _normalize_paths(cfg: RenderConfig, base_dir: Path) -> RenderConfig:
    cfg.source.source_images = _resolve_path(_maybe_path(cfg.source.source_images), base_dir)
    cfg.source.source_latents = _resolve_path(_maybe_path(cfg.source.source_latents), base_dir)
    cfg.projection.latent_dir = _resolve_path(_maybe_path(cfg.projection.latent_dir), base_dir)
    cfg.output.output_dir = _resolve_path(Path(cfg.output.output_dir), base_dir)
    return cfg


def load_render_config(path: Path | None = None) -> RenderConfig:
    cfg = RenderConfig()
    if path is None:
        return _normalize_paths(cfg, Path.cwd())

    path = Path(path).resolve()
    raw_data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw_data, dict):
        raise ValueError("Top-level YAML config must be a mapping")
    _update_dataclass(cfg, raw_data)
    return _normalize_paths(cfg, path.parent)
