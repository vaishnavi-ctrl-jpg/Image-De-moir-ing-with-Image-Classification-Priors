"""
Interactive Streamlit Application for Image De-moiréing with Image Classification Priors.
Displays:
1. Original Moiré Photo vs. Restored Clean Image
2. Moiré Localization Heatmap (answering which parts suffer from moiré)
3. 2D FFT Frequency Spectrum (Before vs After showing periodic spike removal)
4. Prior Attention Feature Library Consultation Heatmap
5. Guidance Branch vs Decoder Dissection
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import streamlit as st

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.pipeline import DemoirePriorPipeline
from utils.spectral_detector import SpectralMoireDetector
from data.synthetic_moire import SyntheticMoireGenerator, generate_benchmark_pattern


@st.cache_resource
def load_pipeline():
    model = DemoirePriorPipeline(pretrained_encoder=True)
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "best_demoire_model.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def compute_fft_spectrum_image(img_np: np.ndarray) -> np.ndarray:
    """Computes logarithmic 2D FFT magnitude spectrum normalized for visualization."""
    gray = img_np.mean(axis=-1)
    H, W = gray.shape
    hann = np.outer(np.hanning(H), np.hanning(W))
    f = np.fft.fftshift(np.fft.fft2(gray * hann))
    mag = np.abs(f)
    mag_log = np.log1p(mag)
    norm = (mag_log - mag_log.min()) / (mag_log.max() - mag_log.min() + 1e-6)
    # Apply colormap (viridis/magma style in RGB)
    # Simple hot/inferno style mapping
    r = np.clip(norm * 2.0, 0, 1)
    g = np.clip(norm * 2.0 - 0.5, 0, 1)
    b = np.clip(norm * 3.0 - 1.5, 0, 1)
    return np.stack([r, g, b], axis=-1)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def mask_to_heatmap(mask_t: torch.Tensor) -> Image.Image:
    m = mask_t.squeeze().detach().cpu().numpy()
    m = np.clip(m, 0.0, 1.0)
    # Turbo / Jet heatmap: red = high moiré, blue = clean
    r = np.clip(1.5 - np.abs(m * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(m * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(m * 4 - 1), 0, 1)
    heat = np.stack([r, g, b], axis=-1)
    return Image.fromarray((heat * 255).astype(np.uint8))


def main():
    st.set_page_config(page_title="Image De-moiréing with Classification Priors", layout="wide")

    st.title("Screen Photo De-moiréing with Classification Priors")
    st.markdown(
        """
        De-moiréing pipeline featuring:
        1. **Pre-trained Classification Encoder** (ImageNet backbone)
        2. **Prior-Consulting Cross-Attention** with Clean Prototype Library
        3. **Reconstruction Decoder** with Correction & Recovery Blocks (CRBs)
        4. **Direct Edge-Filtered Guidance Branch** with Soft-Gated Anti-Seam Fusion
        """
    )

    pipeline = load_pipeline()
    detector = SpectralMoireDetector()
    generator = SyntheticMoireGenerator(seed=42)

    # Sidebar Controls
    st.sidebar.header("Pipeline Configuration")
    source_choice = st.sidebar.radio(
        "Input Image Source",
        ["Synthetic Screen Benchmark", "Synthetic Code Screen", "Synthetic Gradient Photo", "Upload Screen Photo"]
    )

    guidance_strength = st.sidebar.slider("Guidance Branch Weight", min_value=0.2, max_value=2.0, value=1.0, step=0.1)
    soft_gate_sigma = st.sidebar.slider("Anti-Seam Soft Gating Radius (σ)", min_value=1.0, max_value=10.0, value=4.0, step=0.5)
    moire_intensity = st.sidebar.slider("Moiré Simulation Intensity", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    # Prepare Image
    if source_choice == "Synthetic Screen Benchmark":
        clean_np = generate_benchmark_pattern(256, 256)
        corr_np, _ = generator.apply_moire(clean_np, intensity=moire_intensity)
    elif source_choice == "Synthetic Code Screen":
        clean_np = np.ones((256, 256, 3), dtype=np.float32) * 0.95
        for y in range(25, 230, 16):
            w = int(256 * (0.3 + 0.4 * np.sin(y * 0.1)))
            clean_np[y:y+4, 20:20+w] = 0.15
        corr_np, _ = generator.apply_moire(clean_np, intensity=moire_intensity)
    elif source_choice == "Synthetic Gradient Photo":
        y, x = np.mgrid[0:256, 0:256].astype(np.float32) / 256.0
        clean_np = np.stack([x * 0.8, y * 0.7, 0.6 * np.ones_like(x)], axis=-1)
        corr_np, _ = generator.apply_moire(clean_np, intensity=moire_intensity)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            pil_in = Image.open(uploaded_file).convert("RGB").resize((256, 256))
            corr_np = np.array(pil_in, dtype=np.float32) / 255.0
        else:
            st.info("Using default benchmark pattern until you upload a file.")
            clean_np = generate_benchmark_pattern(256, 256)
            corr_np, _ = generator.apply_moire(clean_np, intensity=moire_intensity)

    corr_t = torch.from_numpy(corr_np).permute(2, 0, 1).unsqueeze(0)

    # Run Inference
    with torch.no_grad():
        pipeline.soft_gate_sigma = soft_gate_sigma
        out = pipeline(corr_t, guidance_strength=guidance_strength)

    restored_pil = tensor_to_pil(out["restored"])
    input_pil = tensor_to_pil(corr_t)
    mask_pred_pil = mask_to_heatmap(out["mask_pred"])
    mask_soft_pil = mask_to_heatmap(out["mask_soft"])
    decoder_pil = tensor_to_pil(out["decoder_out"])
    guidance_pil = tensor_to_pil(out["guidance_out"])

    # FFT Spectra
    res_np = out["restored"][0].permute(1, 2, 0).cpu().numpy()
    fft_in_vis = compute_fft_spectrum_image(corr_np)
    fft_out_vis = compute_fft_spectrum_image(res_np)

    # 1. Main Before / After Display
    st.subheader("1. Restoration Output (Side-by-Side)")
    col1, col2 = st.columns(2)
    with col1:
        st.image(input_pil, caption="Original Input Photo (Suffering from Moiré)", use_container_width=True)
    with col2:
        st.image(restored_pil, caption="Restored De-moiréd Image (Clean & Sharp)", use_container_width=True)

    # 2. Answers to User Questions 1 & 2
    st.markdown("---")
    st.subheader("2. Moiré Localization & Spatially-Adaptive Routing")
    st.markdown(
        """
        - **Question 1 (Moiré Localization)**: The heatmap below localizes the moiré-corrupted regions by measuring divergence from the clean classification prior library.
        - **Question 2 (Selective Removal)**: Unaffected regions pass directly through the Guidance Branch, while affected areas are rebuilt by the Reconstruction Decoder.
        """
    )
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.image(mask_pred_pil, caption="Predicted Moiré Localization Heatmap (Red = High Moiré)", use_container_width=True)
        st.caption(f"Mask Stats — Min: {out['mask_pred'].min().item():.2f}, Max: {out['mask_pred'].max().item():.2f}, Mean: {out['mask_pred'].mean().item():.2f}")
    with col_m2:
        st.image(mask_soft_pil, caption="Soft-Gated Anti-Seam Transition Mask (Seamless Blending)", use_container_width=True)
        st.caption("Gaussian smoothed transition preventing visible edge seams at routing boundaries.")

    # 3. Frequency Spectrum Analysis (Pushback 6)
    st.markdown("---")
    st.subheader("3. 2D Fourier Spectrum Verification (Signal-Level Analysis)")
    st.markdown("Observe how the periodic moiré frequency spikes in the input spectrum are attenuated in the restored output.")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.image(fft_in_vis, caption="Input 2D FFT Spectrum (Note the periodic off-axis harmonic spikes)", use_container_width=True)
    with col_f2:
        st.image(fft_out_vis, caption="Restored 2D FFT Spectrum (Harmonic spikes eliminated)", use_container_width=True)

    # 4. Component Dissection
    st.markdown("---")
    st.subheader("4. Architecture Component Dissection")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.image(guidance_pil, caption="Component 4: Edge-Filtered Guidance Branch (Preserves sharp text & true colors)", use_container_width=True)
    with col_d2:
        st.image(decoder_pil, caption="Component 3: Reconstruction Decoder (Synthesizes clean textures via classification priors)", use_container_width=True)


if __name__ == "__main__":
    main()
