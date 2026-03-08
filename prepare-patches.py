from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL

from source_latents import DEFAULT_VAE_ID, prepare_latents_from_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode a folder of images into SDXL VAE latents."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing source images.")
    parser.add_argument("output_dir", type=Path, help="Folder to save latent files into.")
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Optional maximum encoded width. Preserves aspect ratio.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=None,
        help="Optional maximum encoded height. Preserves aspect ratio.",
    )
    parser.add_argument(
        "--mode",
        choices=("mean", "sample"),
        default="mean",
        help="Use posterior mean or sample when encoding.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to run the VAE on, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images inside the input folder.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip output files that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_width is not None and args.max_width < 8:
        raise ValueError("--max-width must be at least 8 pixels.")
    if args.max_height is not None and args.max_height < 8:
        raise ValueError("--max-height must be at least 8 pixels.")

    print(f"Loading VAE: {DEFAULT_VAE_ID}")
    vae = AutoencoderKL.from_pretrained(DEFAULT_VAE_ID, torch_dtype=torch.float16).to(args.device)
    vae.eval()

    latent_paths = prepare_latents_from_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vae=vae,
        device=args.device,
        max_width=args.max_width,
        max_height=args.max_height,
        mode=args.mode,
        recursive=args.recursive,
        skip_existing=args.skip_existing,
        vae_id=DEFAULT_VAE_ID,
    )
    print(f"Saved or reused {len(latent_paths)} latent file(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
