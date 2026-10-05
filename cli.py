"""
Command Line Interface (CLI) for Image De-moiréing Pipeline.
Usage:
  python cli.py --input path/to/moire_photo.png --output path/to/cleaned.png
  python cli.py --benchmark
"""

import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models.pipeline import DemoirePriorPipeline
from data.synthetic_moire import SyntheticMoireGenerator, generate_benchmark_pattern


def run_cli():
    parser = argparse.ArgumentParser(description="Image De-moiréing with Classification Priors")
    parser.add_argument("--input", type=str, help="Path to input moiré image")
    parser.add_argument("--output", type=str, default="cleaned_output.png", help="Path to save restored image")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark test on synthetic pattern")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_demoire_model.pth", help="Path to model weights")
    args = parser.parse_args()

    model = DemoirePriorPipeline(pretrained_encoder=True)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded weights from {args.checkpoint}")
    model.eval()

    if args.benchmark or not args.input:
        print("Running benchmark inference on synthetic screen moiré pattern...")
        gen = SyntheticMoireGenerator(seed=42)
        clean = generate_benchmark_pattern(256, 256)
        corr, mask = gen.apply_moire(clean, intensity=0.5)
        corr_t = torch.from_numpy(corr).permute(2, 0, 1).unsqueeze(0)
    else:
        pil_img = Image.open(args.input).convert("RGB")
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        corr_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        out = model(corr_t)
        res_t = out["restored"]

    res_np = res_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    res_pil = Image.fromarray((np.clip(res_np, 0.0, 1.0) * 255).astype(np.uint8))
    res_pil.save(args.output)
    print(f"Clean restored image saved to: {args.output}")


if __name__ == "__main__":
    run_cli()
