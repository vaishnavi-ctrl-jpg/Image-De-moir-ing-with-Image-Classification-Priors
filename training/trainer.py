"""
Training Pipeline for Image De-moiréing Network.
Trains:
1. Prior Attention Module & Clean Codebook
2. Reconstruction Decoder (CRBs)
3. Guidance Branch & Soft-Gated Fusion
4. Supervised Moiré Mask Predictor using Paired Difference Pseudo-Labels + Teacher Detector.
"""

import math
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.pipeline import DemoirePriorPipeline
from utils.spectral_detector import SpectralMoireDetector
from data.synthetic_moire import SyntheticMoireGenerator
from training.losses import DemoireCompositeLoss


class SyntheticMoireDataset(Dataset):
    """
    Dataset generating synthetic paired screen photo moiré images on-the-fly.
    """
    def __init__(self, num_samples: int = 80, height: int = 256, width: int = 256, seed: int = 42):
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.generator = SyntheticMoireGenerator(seed=seed)
        self.teacher = SpectralMoireDetector()
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def _generate_clean_base(self, index: int) -> torch.Tensor:
        """Generates diverse clean screen backgrounds (slides, text, code, photographs)."""
        rng = torch.Generator().manual_seed(self.seed + index)
        mode = index % 4
        H, W = self.height, self.width
        img = torch.ones(3, H, W, dtype=torch.float32)

        if mode == 0:
            # Code / Text screen
            img[:, :, :] = 0.95
            for y in range(20, H - 20, 16):
                w_line = int(torch.randint(40, W - 40, (1,), generator=rng).item())
                img[:, y:y+4, 20:20+w_line] = 0.15
        elif mode == 1:
            # Presentation Slide / UI
            img[0, :50, :] = 0.2
            img[1, :50, :] = 0.4
            img[2, :50, :] = 0.8
            img[:, 80:200, 30:120] = 0.7
            img[:, 80:200, 140:230] = 0.85
        elif mode == 2:
            # Smooth photographic gradient / Wallpaper
            y = torch.linspace(0, 1, H).view(1, H, 1).repeat(1, 1, W)
            x = torch.linspace(0, 1, W).view(1, 1, W).repeat(1, H, 1)
            img[0] = (x + y) * 0.5
            img[1] = (1.0 - x) * 0.7
            img[2] = 0.8 * torch.sin(x * math.pi)
        else:
            # Natural image pattern
            y, x = torch.meshgrid(torch.linspace(-3, 3, H), torch.linspace(-3, 3, W), indexing="ij")
            r = torch.sqrt(x**2 + y**2)
            img[0] = (torch.sin(r * 2) * 0.5 + 0.5)
            img[1] = (torch.cos(x) * 0.5 + 0.5)
            img[2] = 0.6

        return img.clamp(0.0, 1.0)

    def __getitem__(self, idx: int):
        clean_t = self._generate_clean_base(idx)
        clean_np = clean_t.permute(1, 2, 0).numpy()

        # Apply synthetic moiré
        corr_np, synth_mask_np = self.generator.apply_moire(clean_np, intensity=0.45)
        corr_t = torch.from_numpy(corr_np).permute(2, 0, 1)

        # Generate teacher mask using smoothed paired difference
        diff = torch.abs(corr_t - clean_t).mean(dim=0, keepdim=True)
        pad = 7
        pooled = F.avg_pool2d(F.pad(diff.unsqueeze(0), (pad, pad, pad, pad), mode="reflect"), kernel_size=15, stride=1).squeeze(0)
        teacher_mask = (pooled / pooled.max().clamp(min=1e-4)).clamp(0.0, 1.0)

        return corr_t, clean_t, teacher_mask


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def train_model(
    epochs: int = 6,
    batch_size: int = 4,
    lr: float = 3e-4,
    save_dir: str = "checkpoints"
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}")

    dataset = SyntheticMoireDataset(num_samples=60, height=256, width=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DemoirePriorPipeline(pretrained_encoder=True).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    loss_fn = DemoireCompositeLoss(weight_mask=1.0).to(device)

    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    best_psnr = -1.0
    checkpoint_path = os.path.join(save_dir, "best_demoire_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        psnr_list = []

        for step, (corr, clean, mask_teacher) in enumerate(dataloader):
            corr = corr.to(device)
            clean = clean.to(device)
            mask_teacher = mask_teacher.to(device)

            optimizer.zero_grad()
            out = model(corr)
            losses = loss_fn(out, clean, mask_teacher)

            loss = losses["loss_total"]
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            psnr = compute_psnr(out["restored"], clean)
            psnr_list.append(psnr)

        avg_loss = epoch_loss / len(dataloader)
        avg_psnr = sum(psnr_list) / len(psnr_list)

        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} (L1: {losses['loss_l1']:.4f}, FFT: {losses['loss_fft']:.4f}, Mask: {losses['loss_mask']:.4f}) | PSNR: {avg_psnr:.2f} dB")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "psnr": best_psnr,
            }, checkpoint_path)

    print(f"Training complete! Best checkpoint saved to: {checkpoint_path} (PSNR: {best_psnr:.2f} dB)")
    return checkpoint_path


if __name__ == "__main__":
    train_model(epochs=6, batch_size=4, lr=3e-4)
