"""
FLUX.1 [dev] FP8 Downloader for Google Colab Pro
Downloads FLUX.1-dev model, T5/CLIP text encoders, and VAE into Google Drive.
"""

from pathlib import Path
from pipeline_tools import install_model, DRIVE_MASTER_DIR

def download_flux_models():
    print("================================================================")
    print("      Downloading FLUX.1 [dev] FP8 Suite to Google Drive       ")
    print("================================================================")

    # 1. FLUX.1 [dev] UNet (FP8 version ~11.9GB)
    unet_url = "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors"
    print("\n[1/4] FLUX.1-dev Diffusion Model (FP8 ~11.9GB)...")
    install_model(
        url_or_hf_id=unet_url,
        target_dir="models/unet",
        filename="flux1-dev-fp8.safetensors"
    )

    # 2. CLIP-L Text Encoder (~246MB)
    clip_l_url = "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
    print("\n[2/4] CLIP-L Text Encoder (~246MB)...")
    install_model(
        url_or_hf_id=clip_l_url,
        target_dir="models/clip",
        filename="clip_l.safetensors"
    )

    # 3. T5-XXL Text Encoder (FP8 ~4.8GB)
    t5_url = "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors"
    print("\n[3/4] T5-XXL Text Encoder (FP8 ~4.8GB)...")
    install_model(
        url_or_hf_id=t5_url,
        target_dir="models/clip",
        filename="t5xxl_fp8_e4m3fn.safetensors"
    )

    # 4. FLUX VAE (~335MB) - Use public schnell repo (identical binary, no token required)
    vae_url = "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors"
    print("\n[4/4] FLUX VAE (~335MB)...")
    install_model(
        url_or_hf_id=vae_url,
        target_dir="models/vae",
        filename="ae.safetensors"
    )

    print("\n================================================================")
    print(" [SUCCESS] FLUX.1 Suite is ready in Google Drive!")
    print(f" - UNet: {DRIVE_MASTER_DIR / 'models' / 'unet' / 'flux1-dev-fp8.safetensors'}")
    print(f" - CLIP-L: {DRIVE_MASTER_DIR / 'models' / 'clip' / 'clip_l.safetensors'}")
    print(f" - T5-XXL: {DRIVE_MASTER_DIR / 'models' / 'clip' / 't5xxl_fp8_e4m3fn.safetensors'}")
    print(f" - VAE: {DRIVE_MASTER_DIR / 'models' / 'vae' / 'ae.safetensors'}")
    print("================================================================")

if __name__ == "__main__":
    download_flux_models()
