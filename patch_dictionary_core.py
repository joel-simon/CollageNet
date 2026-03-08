from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import felzenszwalb, mark_boundaries, slic


LATENT_DOWNSAMPLE_FACTOR = 8


@dataclass
class SourceImageRecord:
    npz_path: str
    source_path: str
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    processed_to_original_scale: tuple[float, float]
    latent_downsample_factor: int = LATENT_DOWNSAMPLE_FACTOR
    rotation_k: int = 0


@dataclass
class PatchBank:
    raw_patches: torch.Tensor
    normalized_patches: torch.Tensor
    patch_source_record_indices: np.ndarray
    patch_latent_y: np.ndarray
    patch_latent_x: np.ndarray
    patch_original_bbox_xyxy: np.ndarray
    source_records: list[SourceImageRecord] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)
    source_image_cache: dict[str, Image.Image] = field(default_factory=dict)
    source_latent_cache: dict[str, torch.Tensor] = field(default_factory=dict)


def list_latent_files(latent_source: Path) -> list[Path]:
    latent_source = Path(latent_source)
    if latent_source.is_file():
        if latent_source.suffix.lower() != ".npz":
            raise ValueError(f"Expected a .npz latent file, got {latent_source}")
        return [latent_source]
    if not latent_source.exists():
        raise FileNotFoundError(f"Latent source does not exist: {latent_source}")
    if not latent_source.is_dir():
        raise ValueError(f"Latent source must be a .npz file or directory: {latent_source}")

    files = sorted(latent_source.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No latent files found in {latent_source}")
    return files


def read_npz_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def load_latent_record(path: Path) -> tuple[torch.Tensor, SourceImageRecord]:
    data = np.load(path)
    latent = torch.from_numpy(data["latent"].astype(np.float32))
    processed_size = (
        int(read_npz_scalar(data["processed_width"])),
        int(read_npz_scalar(data["processed_height"])),
    )
    original_size = (
        int(read_npz_scalar(data["original_width"])),
        int(read_npz_scalar(data["original_height"])),
    )
    scale_x = float(
        read_npz_scalar(
            data.get("processed_to_original_scale_x", original_size[0] / processed_size[0])
        )
    )
    scale_y = float(
        read_npz_scalar(
            data.get("processed_to_original_scale_y", original_size[1] / processed_size[1])
        )
    )
    latent_downsample_factor = int(
        read_npz_scalar(data.get("latent_downsample_factor", LATENT_DOWNSAMPLE_FACTOR))
    )
    record = SourceImageRecord(
        npz_path=str(path),
        source_path=str(read_npz_scalar(data["source_path"])),
        original_size=original_size,
        processed_size=processed_size,
        processed_to_original_scale=(scale_x, scale_y),
        latent_downsample_factor=latent_downsample_factor,
    )
    return latent, record


def rotate_latent_and_record(
    latent: torch.Tensor,
    record: SourceImageRecord,
    rotation_k: int,
) -> tuple[torch.Tensor, SourceImageRecord]:
    rotation_k = int(rotation_k) % 4
    if rotation_k == 0:
        return latent, record

    rotated_latent = torch.rot90(latent, k=rotation_k, dims=(-2, -1)).contiguous()

    if rotation_k % 2 == 1:
        original_size = (record.original_size[1], record.original_size[0])
        processed_size = (record.processed_size[1], record.processed_size[0])
        processed_to_original_scale = (
            record.processed_to_original_scale[1],
            record.processed_to_original_scale[0],
        )
    else:
        original_size = record.original_size
        processed_size = record.processed_size
        processed_to_original_scale = record.processed_to_original_scale

    rotated_record = SourceImageRecord(
        npz_path=record.npz_path,
        source_path=record.source_path,
        original_size=original_size,
        processed_size=processed_size,
        processed_to_original_scale=processed_to_original_scale,
        latent_downsample_factor=record.latent_downsample_factor,
        rotation_k=rotation_k,
    )
    return rotated_latent, rotated_record


def collect_patch_positions(height: int, width: int, patch_size: int) -> list[tuple[int, int]]:
    if height < patch_size or width < patch_size:
        return []
    return [
        (y, x)
        for y in range(0, height - patch_size + 1, patch_size)
        for x in range(0, width - patch_size + 1, patch_size)
    ]


def latent_patch_to_processed_bbox(
    latent_y: int,
    latent_x: int,
    patch_size: int,
    latent_downsample_factor: int,
) -> tuple[int, int, int, int]:
    left = latent_x * latent_downsample_factor
    top = latent_y * latent_downsample_factor
    right = (latent_x + patch_size) * latent_downsample_factor
    bottom = (latent_y + patch_size) * latent_downsample_factor
    return left, top, right, bottom


def processed_bbox_to_original_bbox(
    processed_bbox: tuple[int, int, int, int],
    record: SourceImageRecord,
) -> tuple[int, int, int, int]:
    scale_x, scale_y = record.processed_to_original_scale
    left = max(0, min(record.original_size[0], int(round(processed_bbox[0] * scale_x))))
    top = max(0, min(record.original_size[1], int(round(processed_bbox[1] * scale_y))))
    right = max(left + 1, min(record.original_size[0], int(round(processed_bbox[2] * scale_x))))
    bottom = max(top + 1, min(record.original_size[1], int(round(processed_bbox[3] * scale_y))))
    return left, top, right, bottom


def sample_positions_evenly(
    source_specs: list[tuple[Path, int]],
    patch_size: int,
    total_patches: int,
    seed: int,
) -> dict[tuple[Path, int], list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    shuffled_positions: dict[tuple[Path, int], list[tuple[int, int]]] = {}

    for path, rotation_k in source_specs:
        latent, record = load_latent_record(path)
        latent, _ = rotate_latent_and_record(latent, record, rotation_k)
        _, height, width = latent.shape
        positions = collect_patch_positions(height, width, patch_size)
        if positions:
            rng.shuffle(positions)
        shuffled_positions[(path, rotation_k)] = positions

    total_available = sum(len(positions) for positions in shuffled_positions.values())
    if total_available <= total_patches:
        return shuffled_positions

    allocations = {source_spec: [] for source_spec in source_specs}
    remaining = total_patches
    active_files = [source_spec for source_spec in source_specs if shuffled_positions[source_spec]]

    while remaining > 0 and active_files:
        per_file = max(1, remaining // len(active_files))
        next_active_files: list[tuple[Path, int]] = []

        for source_spec in active_files:
            available = len(shuffled_positions[source_spec]) - len(allocations[source_spec])
            take = min(available, per_file)
            start = len(allocations[source_spec])
            end = start + take
            allocations[source_spec].extend(shuffled_positions[source_spec][start:end])
            remaining -= take

            if len(allocations[source_spec]) < len(shuffled_positions[source_spec]):
                next_active_files.append(source_spec)

            if remaining <= 0:
                break

        active_files = next_active_files

    if remaining > 0:
        for source_spec in source_specs:
            available = len(shuffled_positions[source_spec]) - len(allocations[source_spec])
            if available <= 0:
                continue
            start = len(allocations[source_spec])
            end = start + min(available, remaining)
            allocations[source_spec].extend(shuffled_positions[source_spec][start:end])
            remaining -= end - start
            if remaining <= 0:
                break

    return allocations


def normalize_patch_vectors(vectors: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    centered = vectors - vectors.mean(dim=1, keepdim=True)
    norms = centered.norm(dim=1, keepdim=True).clamp_min(eps)
    return centered / norms


def build_patch_bank(cfg: Any, device: str, patch_bank_dtype: torch.dtype) -> PatchBank:
    files = list_latent_files(Path(cfg.latent_dir))
    rotation_ks = [0, 1, 2, 3] if bool(getattr(cfg, "do_rotated", False)) else [0]
    source_specs = [(path, rotation_k) for path in files for rotation_k in rotation_ks]
    allocations = sample_positions_evenly(
        source_specs=source_specs,
        patch_size=cfg.patch_size,
        total_patches=cfg.total_patches,
        seed=cfg.random_seed,
    )

    patch_vectors: list[torch.Tensor] = []
    source_counts: dict[str, int] = {}
    source_records: list[SourceImageRecord] = []
    patch_source_record_indices: list[int] = []
    patch_latent_y: list[int] = []
    patch_latent_x: list[int] = []
    patch_original_bbox_xyxy: list[tuple[int, int, int, int]] = []

    for path, rotation_k in source_specs:
        latent, base_record = load_latent_record(path)
        latent, record = rotate_latent_and_record(latent, base_record, rotation_k)
        source_record_index = len(source_records)
        source_records.append(record)
        positions = allocations[(path, rotation_k)]
        if not positions:
            continue

        selected = []
        for y, x in positions:
            patch = latent[:, y : y + cfg.patch_size, x : x + cfg.patch_size]
            selected.append(patch.reshape(-1))
            processed_bbox = latent_patch_to_processed_bbox(
                latent_y=y,
                latent_x=x,
                patch_size=cfg.patch_size,
                latent_downsample_factor=record.latent_downsample_factor,
            )
            patch_source_record_indices.append(source_record_index)
            patch_latent_y.append(y)
            patch_latent_x.append(x)
            patch_original_bbox_xyxy.append(processed_bbox_to_original_bbox(processed_bbox, record))

        if not selected:
            continue

        stacked = torch.stack(selected, dim=0)
        patch_vectors.append(stacked)
        source_name = path.name if rotation_k == 0 else f"{path.stem}_rot{rotation_k * 90}{path.suffix}"
        source_counts[source_name] = stacked.shape[0]

    if not patch_vectors:
        raise RuntimeError("Could not extract any patches from the latent bank")

    raw = torch.cat(patch_vectors, dim=0).to(device=device, dtype=patch_bank_dtype).contiguous()
    normalized = normalize_patch_vectors(raw.float()).contiguous()

    return PatchBank(
        raw_patches=raw,
        normalized_patches=normalized,
        patch_source_record_indices=np.asarray(patch_source_record_indices, dtype=np.int32),
        patch_latent_y=np.asarray(patch_latent_y, dtype=np.int32),
        patch_latent_x=np.asarray(patch_latent_x, dtype=np.int32),
        patch_original_bbox_xyxy=np.asarray(patch_original_bbox_xyxy, dtype=np.int32),
        source_records=source_records,
        source_counts=source_counts,
    )


def chunked_topk_cosine(
    query_vectors: torch.Tensor,
    dictionary_vectors: torch.Tensor,
    top_k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dictionary_vectors.shape[0] < top_k:
        raise ValueError(
            f"Dictionary only has {dictionary_vectors.shape[0]} patches, but top_k={top_k}"
        )

    best_scores = None
    best_indices = None

    for start in range(0, dictionary_vectors.shape[0], chunk_size):
        chunk = dictionary_vectors[start : start + chunk_size]
        scores = query_vectors @ chunk.T
        chunk_scores, chunk_local_idx = torch.topk(scores, k=min(top_k, chunk.shape[0]), dim=1)
        chunk_indices = chunk_local_idx + start

        if best_scores is None:
            best_scores = chunk_scores
            best_indices = chunk_indices
            continue

        merged_scores = torch.cat([best_scores, chunk_scores], dim=1)
        merged_indices = torch.cat([best_indices, chunk_indices], dim=1)
        keep_scores, keep_idx = torch.topk(merged_scores, k=top_k, dim=1)
        best_scores = keep_scores
        best_indices = torch.gather(merged_indices, 1, keep_idx)

    return best_scores, best_indices


def unfold_latent_patches(latents: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Latent size {(height, width)} must be divisible by patch_size={patch_size}"
        )
    unfolded = F.unfold(latents, kernel_size=patch_size, stride=patch_size)
    return unfolded.transpose(1, 2).reshape(batch, -1, channels * patch_size * patch_size)


def fold_latent_patches(
    patch_vectors: torch.Tensor,
    output_size: tuple[int, int],
    patch_size: int,
) -> torch.Tensor:
    batch, num_patches, patch_dim = patch_vectors.shape
    folded = F.fold(
        patch_vectors.reshape(batch, num_patches, patch_dim).transpose(1, 2),
        output_size=output_size,
        kernel_size=patch_size,
        stride=patch_size,
    )
    return folded


def decode_latents(pipe: Any, latents: torch.Tensor) -> Image.Image:
    latents = latents.to(device=pipe.device, dtype=pipe.vae.dtype)
    latents = latents / pipe.vae.config.scaling_factor
    with torch.inference_mode():
        image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].detach().cpu().permute(1, 2, 0).float().numpy()
    image = (image * 255.0).round().astype(np.uint8)
    return Image.fromarray(image)


def load_source_image(record: SourceImageRecord, cache: dict[str, Image.Image]) -> Image.Image:
    cache_key = f"{record.source_path}::rot{record.rotation_k}"
    if cache_key not in cache:
        with Image.open(record.source_path) as image:
            converted = image.convert("RGB")
            if record.rotation_k:
                converted = converted.rotate(90 * record.rotation_k, expand=True)
            cache[cache_key] = converted.copy()
    return cache[cache_key]


def load_source_latent(record: SourceImageRecord, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    cache_key = f"{record.npz_path}::rot{record.rotation_k}"
    if cache_key not in cache:
        latent, _ = load_latent_record(Path(record.npz_path))
        latent, _ = rotate_latent_and_record(latent, record, record.rotation_k)
        cache[cache_key] = latent
    return cache[cache_key]


def reference_window_is_possible(
    patch_bank: PatchBank,
    window_height: int,
    window_width: int,
) -> bool:
    for record in patch_bank.source_records:
        latent = load_source_latent(record, patch_bank.source_latent_cache)
        _, height, width = latent.shape
        if height >= window_height and width >= window_width:
            return True
    return False


def latent_window_to_original_bbox(
    latent_y: int,
    latent_x: int,
    latent_height: int,
    latent_width: int,
    record: SourceImageRecord,
) -> tuple[int, int, int, int]:
    processed_bbox = (
        latent_x * record.latent_downsample_factor,
        latent_y * record.latent_downsample_factor,
        (latent_x + latent_width) * record.latent_downsample_factor,
        (latent_y + latent_height) * record.latent_downsample_factor,
    )
    return processed_bbox_to_original_bbox(processed_bbox, record)


def render_pixel_collage_from_assignments(
    patch_bank: PatchBank,
    selected_patch_indices: np.ndarray,
    grid_shape: tuple[int, int],
    patch_size: int,
    pixel_render_scale: int = 1,
) -> Image.Image:
    if pixel_render_scale < 1:
        raise ValueError("pixel_render_scale must be >= 1")

    grid_height, grid_width = grid_shape
    patch_indices = np.asarray(selected_patch_indices, dtype=np.int32).reshape(-1)
    patch_pixel_size = patch_size * LATENT_DOWNSAMPLE_FACTOR * pixel_render_scale
    canvas = Image.new("RGB", (grid_width * patch_pixel_size, grid_height * patch_pixel_size))

    for patch_offset, bank_patch_index in enumerate(patch_indices.tolist()):
        source_record_index = int(patch_bank.patch_source_record_indices[bank_patch_index])
        record = patch_bank.source_records[source_record_index]
        source_image = load_source_image(record, patch_bank.source_image_cache)
        bbox = tuple(int(v) for v in patch_bank.patch_original_bbox_xyxy[bank_patch_index])
        crop = source_image.crop(bbox)
        if crop.size != (patch_pixel_size, patch_pixel_size):
            crop = crop.resize((patch_pixel_size, patch_pixel_size), Image.Resampling.LANCZOS)

        row = patch_offset // grid_width
        col = patch_offset % grid_width
        canvas.paste(crop, (col * patch_pixel_size, row * patch_pixel_size))

    return canvas


def render_pixel_collage_from_region_assignments(
    patch_bank: PatchBank,
    region_assignments: list[dict[str, Any]],
    latent_canvas_shape: tuple[int, int],
    pixel_render_scale: int = 1,
) -> Image.Image:
    if pixel_render_scale < 1:
        raise ValueError("pixel_render_scale must be >= 1")

    latent_height, latent_width = latent_canvas_shape
    canvas_width = latent_width * LATENT_DOWNSAMPLE_FACTOR * pixel_render_scale
    canvas_height = latent_height * LATENT_DOWNSAMPLE_FACTOR * pixel_render_scale
    canvas = Image.new("RGB", (canvas_width, canvas_height))

    for assignment in region_assignments:
        source_record = patch_bank.source_records[int(assignment["source_record_index"])]
        source_image = load_source_image(source_record, patch_bank.source_image_cache)
        source_bbox = latent_window_to_original_bbox(
            latent_y=int(assignment["source_latent_y"]),
            latent_x=int(assignment["source_latent_x"]),
            latent_height=int(assignment["bbox_h"]),
            latent_width=int(assignment["bbox_w"]),
            record=source_record,
        )
        crop = source_image.crop(source_bbox)

        target_left = int(assignment["target_x0"]) * LATENT_DOWNSAMPLE_FACTOR * pixel_render_scale
        target_top = int(assignment["target_y0"]) * LATENT_DOWNSAMPLE_FACTOR * pixel_render_scale
        target_width = int(assignment["bbox_w"]) * LATENT_DOWNSAMPLE_FACTOR * pixel_render_scale
        target_height = int(assignment["bbox_h"]) * LATENT_DOWNSAMPLE_FACTOR * pixel_render_scale

        if crop.size != (target_width, target_height):
            crop = crop.resize((target_width, target_height), Image.Resampling.LANCZOS)

        mask = np.asarray(assignment["mask"], dtype=np.uint8) * 255
        mask_image = Image.fromarray(mask, mode="L").resize(
            (target_width, target_height),
            Image.Resampling.NEAREST,
        )
        canvas.paste(crop, (target_left, target_top), mask_image)

    return canvas


def latent_labels_to_debug_image(
    labels: np.ndarray,
    seed: int = 0,
    draw_boundaries: bool = False,
) -> Image.Image:
    labels = np.asarray(labels, dtype=np.int32)
    rng = np.random.default_rng(seed)
    num_labels = int(labels.max()) + 1
    palette = rng.integers(0, 255, size=(max(num_labels, 1), 3), dtype=np.uint8)
    colored = palette[labels]
    if not draw_boundaries:
        return Image.fromarray(colored)

    with_boundaries = mark_boundaries(
        colored.astype(np.float32) / 255.0,
        labels,
        color=(0.0, 0.0, 0.0),
        mode="thick",
    )
    debug_image = (with_boundaries * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(debug_image)


def latent_features_for_slic(latents: torch.Tensor) -> np.ndarray:
    feature_map = latents[0].detach().float().cpu().permute(1, 2, 0).numpy()
    mean = feature_map.mean(axis=(0, 1), keepdims=True)
    std = feature_map.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (feature_map - mean) / std


def latent_unit_features(latents: torch.Tensor) -> np.ndarray:
    feature_map = latents[0].detach().float().cpu().permute(1, 2, 0).numpy()
    norms = np.linalg.norm(feature_map, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    return feature_map / norms


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        rank = self.rank
        parent = self.parent
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1


def labels_from_similarity_threshold(
    features: np.ndarray,
    threshold: float,
    connectivity: int = 4,
) -> np.ndarray:
    height, width, _ = features.shape
    uf = UnionFind(height * width)

    def index(y: int, x: int) -> int:
        return y * width + x

    right_sims = np.sum(features[:, :-1] * features[:, 1:], axis=-1)
    down_sims = np.sum(features[:-1, :] * features[1:, :], axis=-1)

    ys, xs = np.where(right_sims >= threshold)
    for y, x in zip(ys.tolist(), xs.tolist()):
        uf.union(index(y, x), index(y, x + 1))

    ys, xs = np.where(down_sims >= threshold)
    for y, x in zip(ys.tolist(), xs.tolist()):
        uf.union(index(y, x), index(y + 1, x))

    if connectivity == 8:
        down_right_sims = np.sum(features[:-1, :-1] * features[1:, 1:], axis=-1)
        down_left_sims = np.sum(features[:-1, 1:] * features[1:, :-1], axis=-1)

        ys, xs = np.where(down_right_sims >= threshold)
        for y, x in zip(ys.tolist(), xs.tolist()):
            uf.union(index(y, x), index(y + 1, x + 1))

        ys, xs = np.where(down_left_sims >= threshold)
        for y, x in zip(ys.tolist(), xs.tolist()):
            uf.union(index(y, x + 1), index(y + 1, x))

    labels = np.empty((height, width), dtype=np.int32)
    root_to_label: dict[int, int] = {}
    next_label = 0
    for y in range(height):
        for x in range(width):
            root = uf.find(index(y, x))
            if root not in root_to_label:
                root_to_label[root] = next_label
                next_label += 1
            labels[y, x] = root_to_label[root]
    return labels


def sample_random_reference_windows(
    patch_bank: PatchBank,
    window_height: int,
    window_width: int,
    num_candidates: int,
    rng: np.random.Generator,
    device: str,
) -> tuple[torch.Tensor, np.ndarray]:
    eligible_source_indices = []
    for source_index, record in enumerate(patch_bank.source_records):
        latent = load_source_latent(record, patch_bank.source_latent_cache)
        _, height, width = latent.shape
        if height >= window_height and width >= window_width:
            eligible_source_indices.append(source_index)

    if not eligible_source_indices:
        raise RuntimeError(
            f"No reference latents are large enough for window {(window_height, window_width)}"
        )

    windows = []
    metadata: list[tuple[int, int, int]] = []
    for _ in range(num_candidates):
        source_index = int(rng.choice(eligible_source_indices))
        record = patch_bank.source_records[source_index]
        latent = load_source_latent(record, patch_bank.source_latent_cache)
        _, height, width = latent.shape
        max_y = height - window_height
        max_x = width - window_width
        window_y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        window_x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        window = latent[:, window_y : window_y + window_height, window_x : window_x + window_width]
        windows.append(window)
        metadata.append((source_index, window_y, window_x))

    stacked = torch.stack(windows, dim=0).to(device=device, dtype=torch.float32)
    return stacked, np.asarray(metadata, dtype=np.int32)


class DiffusionPatchProjector:
    def __init__(self, cfg: Any, patch_bank: PatchBank):
        self.cfg = cfg
        self.patch_bank = patch_bank
        self.projection_events: list[dict[str, Any]] = []
        self.preview_latents: list[dict[str, Any]] = []
        self.final_selected_patch_indices: torch.Tensor | None = None
        self.final_assignment_grid_shape: tuple[int, int] | None = None
        self.final_region_assignments: list[dict[str, Any]] | None = None
        self.debug_output_dir: Path | None = None

    def set_debug_output_dir(self, path: str | Path | None) -> None:
        self.debug_output_dir = None if path is None else Path(path)
        if self.debug_output_dir is not None:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)

    def should_project(self, step_index: int, total_steps: int) -> bool:
        if self.cfg.projection_every_n_steps <= 0:
            return False
        step_frac = step_index / max(total_steps - 1, 1)
        within_window = self.cfg.projection_start_frac <= step_frac <= self.cfg.projection_end_frac
        on_stride = step_index % self.cfg.projection_every_n_steps == 0
        return within_window and on_stride

    def alpha_for_step(self, step_index: int, total_steps: int) -> float:
        step_frac = step_index / max(total_steps - 1, 1)
        start = self.cfg.projection_start_frac
        end = max(self.cfg.projection_end_frac, start + 1e-6)
        ramp = (step_frac - start) / (end - start)
        ramp = float(min(max(ramp, 0.0), 1.0))
        return (1.0 - ramp) * self.cfg.alpha_start + ramp * self.cfg.alpha_end

    def project_latents(
        self,
        latents: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor, tuple[int, int]]:
        original_dtype = latents.dtype
        batch, _, height, width = latents.shape
        patch_vectors = unfold_latent_patches(latents, self.cfg.patch_size)
        flat_queries = patch_vectors.reshape(-1, patch_vectors.shape[-1]).float()
        normalized_queries = normalize_patch_vectors(flat_queries)

        top_scores, top_indices = chunked_topk_cosine(
            query_vectors=normalized_queries,
            dictionary_vectors=self.patch_bank.normalized_patches,
            top_k=self.cfg.top_k,
            chunk_size=self.cfg.dictionary_chunk_size,
        )

        weights = torch.softmax(top_scores / self.cfg.similarity_temperature, dim=1)
        matched = self.patch_bank.raw_patches[top_indices]
        target = (weights.unsqueeze(-1) * matched.float()).sum(dim=1)
        blended = torch.lerp(flat_queries, target, alpha).to(original_dtype)
        dominant_choice = torch.argmax(weights, dim=1, keepdim=True)
        dominant_indices = torch.gather(top_indices, 1, dominant_choice).squeeze(1)

        reconstructed = fold_latent_patches(
            blended.reshape(batch, -1, blended.shape[-1]),
            output_size=(height, width),
            patch_size=self.cfg.patch_size,
        )
        grid_shape = (height // self.cfg.patch_size, width // self.cfg.patch_size)
        stats = {
            "alpha": float(alpha),
            "mean_top1_cosine": float(top_scores[:, 0].mean().item()),
            "mean_topk_cosine": float(top_scores.mean().item()),
            "num_queries": int(flat_queries.shape[0]),
            "num_unique_selected_patches": int(dominant_indices.unique().numel()),
        }
        return reconstructed, stats, dominant_indices.reshape(batch, -1), grid_shape

    def __call__(self, pipe: Any, step_index: int, timestep: int, callback_kwargs: dict[str, Any]):
        latents = callback_kwargs["latents"]
        total_steps = pipe.num_timesteps

        if not self.should_project(step_index, total_steps):
            return {"latents": latents}

        alpha = self.alpha_for_step(step_index, total_steps)
        projected, stats, selected_patch_indices, grid_shape = self.project_latents(
            latents,
            alpha=alpha,
        )
        self.final_selected_patch_indices = selected_patch_indices.detach().to(
            "cpu",
            dtype=torch.int32,
        )
        self.final_assignment_grid_shape = grid_shape

        event = {
            "step_index": int(step_index),
            "timestep": int(timestep),
            **stats,
        }
        self.projection_events.append(event)

        if self.cfg.preview_every_n_projections > 0:
            if len(self.projection_events) % self.cfg.preview_every_n_projections == 0:
                self.preview_latents.append(
                    {
                        "step_index": int(step_index),
                        "timestep": int(timestep),
                        "latents": projected.detach().to("cpu", dtype=torch.float16),
                    }
                )

        return {"latents": projected}


class LatentSLICRegionProjector(DiffusionPatchProjector):
    def __init__(self, cfg: Any, patch_bank: PatchBank):
        super().__init__(cfg=cfg, patch_bank=patch_bank)
        self.rng = np.random.default_rng(cfg.random_seed)

    def compute_slic_labels(self, latents: torch.Tensor) -> tuple[np.ndarray, float]:
        features = latent_features_for_slic(latents)
        compactness_candidates = [
            float(self.cfg.slic_compactness),
            max(float(self.cfg.slic_compactness), 0.25),
            1.0,
            5.0,
            10.0,
        ]
        best_labels = None
        best_compactness = compactness_candidates[0]
        best_num_labels = -1

        for compactness in compactness_candidates:
            labels = slic(
                features,
                n_segments=int(self.cfg.slic_n_segments),
                compactness=compactness,
                start_label=0,
                channel_axis=-1,
                convert2lab=False,
                enforce_connectivity=True,
            ).astype(np.int32)
            num_labels = int(np.unique(labels).size)
            if num_labels > best_num_labels:
                best_labels = labels
                best_compactness = compactness
                best_num_labels = num_labels
            if num_labels > 1:
                return labels, compactness

        assert best_labels is not None
        return best_labels, best_compactness

    def maybe_save_debug_labels(self, labels: np.ndarray, step_index: int, timestep: int) -> None:
        if self.debug_output_dir is None:
            return
        every_n = max(1, int(getattr(self.cfg, "debug_every_n_projections", 1)))
        if len(self.projection_events) % every_n != 0:
            return
        debug_image = latent_labels_to_debug_image(labels, seed=int(step_index))
        debug_path = self.debug_output_dir / f"slic_step_{step_index:03d}_t{int(timestep):04d}.png"
        debug_image = debug_image.resize(
            (labels.shape[1] * 4, labels.shape[0] * 4),
            Image.Resampling.NEAREST,
        )
        debug_image.save(debug_path)

    def region_ids_for_labels(self, labels: np.ndarray) -> list[int]:
        region_ids = []
        min_area = int(getattr(self.cfg, "region_min_area", 1))
        max_area = int(getattr(self.cfg, "region_max_area", labels.size))
        for region_id in np.unique(labels):
            area = int(np.count_nonzero(labels == region_id))
            if area < min_area or area > max_area:
                continue
            region_ids.append(int(region_id))
        return region_ids

    def region_bbox_is_allowed(self, bbox_h: int, bbox_w: int) -> bool:
        max_bbox_h = int(getattr(self.cfg, "region_max_bbox_h", 0) or 0)
        max_bbox_w = int(getattr(self.cfg, "region_max_bbox_w", 0) or 0)
        if max_bbox_h and bbox_h > max_bbox_h:
            return False
        if max_bbox_w and bbox_w > max_bbox_w:
            return False
        return True

    def score_masked_candidates(
        self,
        target_window: torch.Tensor,
        candidate_windows: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_flat = mask.reshape(-1) > 0.5
        if int(mask_flat.sum().item()) == 0:
            raise RuntimeError("Masked region is empty after bbox extraction")

        target_vec = target_window.reshape(target_window.shape[0], -1)[:, mask_flat].reshape(1, -1)
        candidate_vecs = candidate_windows.reshape(candidate_windows.shape[0], candidate_windows.shape[1], -1)
        candidate_vecs = candidate_vecs[:, :, mask_flat].reshape(candidate_windows.shape[0], -1)
        normalized_target = normalize_patch_vectors(target_vec)
        normalized_candidates = normalize_patch_vectors(candidate_vecs)
        return (normalized_candidates @ normalized_target.T).squeeze(1)

    def project_latents(
        self,
        latents: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, Any]], tuple[int, int], np.ndarray]:
        working = latents.clone()
        _, _, latent_height, latent_width = working.shape
        labels, compactness_used = self.compute_slic_labels(working)
        region_assignments: list[dict[str, Any]] = []
        region_scores: list[float] = []
        region_ids = self.region_ids_for_labels(labels)
        skipped_bbox = 0

        for region_id in region_ids:
            region_mask_np = labels == region_id
            ys, xs = np.where(region_mask_np)
            if ys.size == 0:
                continue

            y0 = int(ys.min())
            y1 = int(ys.max()) + 1
            x0 = int(xs.min())
            x1 = int(xs.max()) + 1
            bbox_h = y1 - y0
            bbox_w = x1 - x0
            if not self.region_bbox_is_allowed(bbox_h, bbox_w):
                skipped_bbox += 1
                continue
            if not reference_window_is_possible(self.patch_bank, bbox_h, bbox_w):
                skipped_bbox += 1
                continue
            target_window = working[0, :, y0:y1, x0:x1].float()
            mask_np = region_mask_np[y0:y1, x0:x1].astype(np.float32)
            mask = torch.from_numpy(mask_np).to(device=working.device, dtype=torch.float32)

            candidate_windows, candidate_metadata = sample_random_reference_windows(
                patch_bank=self.patch_bank,
                window_height=bbox_h,
                window_width=bbox_w,
                num_candidates=int(self.cfg.region_candidate_count),
                rng=self.rng,
                device=working.device,
            )
            scores = self.score_masked_candidates(target_window, candidate_windows, mask)
            best_index = int(torch.argmax(scores).item())
            best_score = float(scores[best_index].item())
            best_window = candidate_windows[best_index]

            mask_3d = mask.unsqueeze(0)
            current = working[0, :, y0:y1, x0:x1]
            blended = current * (1.0 - alpha * mask_3d) + best_window.to(current.dtype) * (alpha * mask_3d)
            working[0, :, y0:y1, x0:x1] = blended

            source_record_index, source_latent_y, source_latent_x = candidate_metadata[best_index].tolist()
            region_assignments.append(
                {
                    "region_id": int(region_id),
                    "source_record_index": int(source_record_index),
                    "source_latent_y": int(source_latent_y),
                    "source_latent_x": int(source_latent_x),
                    "target_y0": y0,
                    "target_x0": x0,
                    "bbox_h": bbox_h,
                    "bbox_w": bbox_w,
                    "mask": mask_np.astype(np.uint8),
                    "score": best_score,
                }
            )
            region_scores.append(best_score)

        stats = {
            "alpha": float(alpha),
            "num_slic_labels": int(np.unique(labels).size),
            "slic_compactness_used": float(compactness_used),
            "num_regions": int(len(region_assignments)),
            "num_regions_skipped_bbox": int(skipped_bbox),
            "mean_region_cosine": float(np.mean(region_scores)) if region_scores else 0.0,
            "max_region_cosine": float(np.max(region_scores)) if region_scores else 0.0,
            "min_region_cosine": float(np.min(region_scores)) if region_scores else 0.0,
        }
        return working, stats, region_assignments, (latent_height, latent_width), labels

    def __call__(self, pipe: Any, step_index: int, timestep: int, callback_kwargs: dict[str, Any]):
        latents = callback_kwargs["latents"]
        total_steps = pipe.num_timesteps

        if not self.should_project(step_index, total_steps):
            return {"latents": latents}

        alpha = self.alpha_for_step(step_index, total_steps)
        projected, stats, region_assignments, latent_shape, labels = self.project_latents(
            latents,
            alpha=alpha,
        )
        self.final_region_assignments = region_assignments if region_assignments else None
        self.final_assignment_grid_shape = latent_shape

        event = {
            "step_index": int(step_index),
            "timestep": int(timestep),
            **stats,
        }
        self.projection_events.append(event)
        self.maybe_save_debug_labels(labels, step_index=step_index, timestep=timestep)

        if self.cfg.preview_every_n_projections > 0:
            if len(self.projection_events) % self.cfg.preview_every_n_projections == 0:
                self.preview_latents.append(
                    {
                        "step_index": int(step_index),
                        "timestep": int(timestep),
                        "latents": projected.detach().to("cpu", dtype=torch.float16),
                    }
                )

        return {"latents": projected}


class LatentThresholdRegionProjector(LatentSLICRegionProjector):
    def __init__(self, cfg: Any, patch_bank: PatchBank):
        super().__init__(cfg=cfg, patch_bank=patch_bank)

    def compute_threshold_labels(self, latents: torch.Tensor) -> tuple[np.ndarray, float]:
        features = latent_unit_features(latents)
        min_regions = int(getattr(self.cfg, "threshold_min_regions", 32))
        max_regions = int(getattr(self.cfg, "threshold_max_regions", 128))
        connectivity = int(getattr(self.cfg, "threshold_connectivity", 4))
        low = float(getattr(self.cfg, "threshold_similarity_low", -0.25))
        high = float(getattr(self.cfg, "threshold_similarity_high", 0.999))
        best_labels = None
        best_threshold = low
        best_distance = float("inf")

        for _ in range(14):
            threshold = (low + high) / 2.0
            labels = labels_from_similarity_threshold(
                features,
                threshold=threshold,
                connectivity=connectivity,
            )
            num_labels = int(np.unique(labels).size)
            if min_regions <= num_labels <= max_regions:
                return labels, threshold

            target_mid = 0.5 * (min_regions + max_regions)
            distance = abs(num_labels - target_mid)
            if distance < best_distance:
                best_distance = distance
                best_labels = labels
                best_threshold = threshold

            if num_labels < min_regions:
                low = threshold
            else:
                high = threshold

        assert best_labels is not None
        return best_labels, best_threshold

    def maybe_save_debug_labels(self, labels: np.ndarray, step_index: int, timestep: int) -> None:
        if self.debug_output_dir is None:
            return
        every_n = max(1, int(getattr(self.cfg, "debug_every_n_projections", 1)))
        if len(self.projection_events) % every_n != 0:
            return
        debug_image = latent_labels_to_debug_image(labels, seed=int(step_index))
        debug_path = self.debug_output_dir / f"threshold_step_{step_index:03d}_t{int(timestep):04d}.png"
        debug_image = debug_image.resize(
            (labels.shape[1] * 4, labels.shape[0] * 4),
            Image.Resampling.NEAREST,
        )
        debug_image.save(debug_path)

    def project_latents(
        self,
        latents: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, Any]], tuple[int, int], np.ndarray]:
        working = latents.clone()
        _, _, latent_height, latent_width = working.shape
        labels, threshold_used = self.compute_threshold_labels(working)
        region_assignments: list[dict[str, Any]] = []
        region_scores: list[float] = []
        region_ids = self.region_ids_for_labels(labels)
        skipped_bbox = 0

        for region_id in region_ids:
            region_mask_np = labels == region_id
            ys, xs = np.where(region_mask_np)
            if ys.size == 0:
                continue

            y0 = int(ys.min())
            y1 = int(ys.max()) + 1
            x0 = int(xs.min())
            x1 = int(xs.max()) + 1
            bbox_h = y1 - y0
            bbox_w = x1 - x0
            if not self.region_bbox_is_allowed(bbox_h, bbox_w):
                skipped_bbox += 1
                continue
            if not reference_window_is_possible(self.patch_bank, bbox_h, bbox_w):
                skipped_bbox += 1
                continue

            target_window = working[0, :, y0:y1, x0:x1].float()
            mask_np = region_mask_np[y0:y1, x0:x1].astype(np.float32)
            mask = torch.from_numpy(mask_np).to(device=working.device, dtype=torch.float32)

            candidate_windows, candidate_metadata = sample_random_reference_windows(
                patch_bank=self.patch_bank,
                window_height=bbox_h,
                window_width=bbox_w,
                num_candidates=int(self.cfg.region_candidate_count),
                rng=self.rng,
                device=working.device,
            )
            scores = self.score_masked_candidates(target_window, candidate_windows, mask)
            best_index = int(torch.argmax(scores).item())
            best_score = float(scores[best_index].item())
            best_window = candidate_windows[best_index]

            mask_3d = mask.unsqueeze(0)
            current = working[0, :, y0:y1, x0:x1]
            blended = current * (1.0 - alpha * mask_3d) + best_window.to(current.dtype) * (alpha * mask_3d)
            working[0, :, y0:y1, x0:x1] = blended

            source_record_index, source_latent_y, source_latent_x = candidate_metadata[best_index].tolist()
            region_assignments.append(
                {
                    "region_id": int(region_id),
                    "source_record_index": int(source_record_index),
                    "source_latent_y": int(source_latent_y),
                    "source_latent_x": int(source_latent_x),
                    "target_y0": y0,
                    "target_x0": x0,
                    "bbox_h": bbox_h,
                    "bbox_w": bbox_w,
                    "mask": mask_np.astype(np.uint8),
                    "score": best_score,
                }
            )
            region_scores.append(best_score)

        stats = {
            "alpha": float(alpha),
            "num_threshold_labels": int(np.unique(labels).size),
            "threshold_similarity_used": float(threshold_used),
            "num_regions": int(len(region_assignments)),
            "num_regions_skipped_bbox": int(skipped_bbox),
            "mean_region_cosine": float(np.mean(region_scores)) if region_scores else 0.0,
            "max_region_cosine": float(np.max(region_scores)) if region_scores else 0.0,
            "min_region_cosine": float(np.min(region_scores)) if region_scores else 0.0,
        }
        return working, stats, region_assignments, (latent_height, latent_width), labels


class LatentFelzenszwalbRegionProjector(LatentSLICRegionProjector):
    def __init__(self, cfg: Any, patch_bank: PatchBank):
        super().__init__(cfg=cfg, patch_bank=patch_bank)

    def compute_felzenszwalb_labels(self, latents: torch.Tensor) -> tuple[np.ndarray, dict[str, float]]:
        features = latent_features_for_slic(latents)
        scale = float(getattr(self.cfg, "felzenszwalb_scale", 1.0))
        sigma = float(getattr(self.cfg, "felzenszwalb_sigma", 0.0))
        min_size = int(getattr(self.cfg, "felzenszwalb_min_size", 20))
        labels = felzenszwalb(
            features,
            scale=scale,
            sigma=sigma,
            min_size=min_size,
            channel_axis=-1,
        ).astype(np.int32)
        return labels, {
            "felzenszwalb_scale_used": scale,
            "felzenszwalb_sigma_used": sigma,
            "felzenszwalb_min_size_used": float(min_size),
        }

    def maybe_save_debug_labels(self, labels: np.ndarray, step_index: int, timestep: int) -> None:
        if self.debug_output_dir is None:
            return
        every_n = max(1, int(getattr(self.cfg, "debug_every_n_projections", 1)))
        if len(self.projection_events) % every_n != 0:
            return
        debug_image = latent_labels_to_debug_image(labels, seed=int(step_index))
        debug_path = self.debug_output_dir / f"felzenszwalb_step_{step_index:03d}_t{int(timestep):04d}.png"
        debug_image = debug_image.resize(
            (labels.shape[1] * 4, labels.shape[0] * 4),
            Image.Resampling.NEAREST,
        )
        debug_image.save(debug_path)

    def project_latents(
        self,
        latents: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, Any]], tuple[int, int], np.ndarray]:
        working = latents.clone()
        _, _, latent_height, latent_width = working.shape
        labels, felz_stats = self.compute_felzenszwalb_labels(working)
        region_assignments: list[dict[str, Any]] = []
        region_scores: list[float] = []
        region_ids = self.region_ids_for_labels(labels)
        skipped_bbox = 0

        for region_id in region_ids:
            region_mask_np = labels == region_id
            ys, xs = np.where(region_mask_np)
            if ys.size == 0:
                continue

            y0 = int(ys.min())
            y1 = int(ys.max()) + 1
            x0 = int(xs.min())
            x1 = int(xs.max()) + 1
            bbox_h = y1 - y0
            bbox_w = x1 - x0
            if not self.region_bbox_is_allowed(bbox_h, bbox_w):
                skipped_bbox += 1
                continue
            if not reference_window_is_possible(self.patch_bank, bbox_h, bbox_w):
                skipped_bbox += 1
                continue

            target_window = working[0, :, y0:y1, x0:x1].float()
            mask_np = region_mask_np[y0:y1, x0:x1].astype(np.float32)
            mask = torch.from_numpy(mask_np).to(device=working.device, dtype=torch.float32)

            candidate_windows, candidate_metadata = sample_random_reference_windows(
                patch_bank=self.patch_bank,
                window_height=bbox_h,
                window_width=bbox_w,
                num_candidates=int(self.cfg.region_candidate_count),
                rng=self.rng,
                device=working.device,
            )
            scores = self.score_masked_candidates(target_window, candidate_windows, mask)
            best_index = int(torch.argmax(scores).item())
            best_score = float(scores[best_index].item())
            best_window = candidate_windows[best_index]

            mask_3d = mask.unsqueeze(0)
            current = working[0, :, y0:y1, x0:x1]
            blended = current * (1.0 - alpha * mask_3d) + best_window.to(current.dtype) * (alpha * mask_3d)
            working[0, :, y0:y1, x0:x1] = blended

            source_record_index, source_latent_y, source_latent_x = candidate_metadata[best_index].tolist()
            region_assignments.append(
                {
                    "region_id": int(region_id),
                    "source_record_index": int(source_record_index),
                    "source_latent_y": int(source_latent_y),
                    "source_latent_x": int(source_latent_x),
                    "target_y0": y0,
                    "target_x0": x0,
                    "bbox_h": bbox_h,
                    "bbox_w": bbox_w,
                    "mask": mask_np.astype(np.uint8),
                    "score": best_score,
                }
            )
            region_scores.append(best_score)

        stats = {
            "alpha": float(alpha),
            "num_felzenszwalb_labels": int(np.unique(labels).size),
            "num_regions": int(len(region_assignments)),
            "num_regions_skipped_bbox": int(skipped_bbox),
            "mean_region_cosine": float(np.mean(region_scores)) if region_scores else 0.0,
            "max_region_cosine": float(np.max(region_scores)) if region_scores else 0.0,
            "min_region_cosine": float(np.min(region_scores)) if region_scores else 0.0,
            **felz_stats,
        }
        return working, stats, region_assignments, (latent_height, latent_width), labels
