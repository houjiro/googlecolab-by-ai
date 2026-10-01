"""
Standalone downloader script for SDXL Base 1.0 & SDXL VAE
"""

from pipeline_tools import install_model, DRIVE_MASTER_DIR

def download_sdxl():
    print("=====================================================")
    print("   Downloading Stable Diffusion XL (Base 1.0) & VAE  ")
    print("=====================================================")

    # 1. SDXL Base 1.0 Checkpoint
    sdxl_url = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    install_model(
        url_or_hf_id=sdxl_url,
        target_dir="models/checkpoints",
        filename="sd_xl_base_1.0.safetensors"
    )

    # 2. SDXL FP16 VAE
    vae_url = "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors"
    install_model(
        url_or_hf_id=vae_url,
        target_dir="models/vae",
        filename="sdxl.vae.safetensors"
    )

    print("\n[SUCCESS] SDXL models are ready in Google Drive:")
    print(f" - Checkpoint: {DRIVE_MASTER_DIR / 'models' / 'checkpoints' / 'sd_xl_base_1.0.safetensors'}")
    print(f" - VAE: {DRIVE_MASTER_DIR / 'models' / 'vae' / 'sdxl.vae.safetensors'}")

if __name__ == "__main__":
    download_sdxl()
