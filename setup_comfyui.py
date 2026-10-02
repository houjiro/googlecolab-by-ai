"""
ComfyUI Master Setup & Environment Manager for Google Colab Pro
Phase 1: Environment Isolation, Acceleration & Caching
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

# --- Configuration ---
TARGET_DRIVE_FOLDER_ID = "1CVwjVWJAR7PDiFbs4IPCoKzE3KYzDk2n"
DRIVE_MOUNT_POINT = Path("/content/drive")
LOCAL_WORK_DIR = Path("/content")
COMFYUI_DIR = LOCAL_WORK_DIR / "ComfyUI"
MINICONDA_DIR = LOCAL_WORK_DIR / "miniconda"
CONDA_ENV_NAME = "comfy_env"

def resolve_target_drive_dir() -> Path:
    """
    Resolve target directory from Google Drive.
    Searches for folder associated with TARGET_DRIVE_FOLDER_ID, or defaults to MyDrive/ComfyUI_Master.
    """
    if not (DRIVE_MOUNT_POINT / "MyDrive").exists():
        return DRIVE_MOUNT_POINT / "MyDrive" / "ComfyUI_Master"

    # Check if folder name can be resolved via PyDrive or Drive search
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
        from google.colab import auth
        from oauth2client.client import GoogleCredentials
        # Attempt silent resolution if auth is already available
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        drive = GoogleDrive(gauth)
        target_folder = drive.CreateFile({'id': TARGET_DRIVE_FOLDER_ID})
        folder_title = target_folder['title']
        print(f"[DRIVE] Target folder identified: '{folder_title}' (ID: {TARGET_DRIVE_FOLDER_ID})")
        # Search in MyDrive
        candidate = DRIVE_MOUNT_POINT / "MyDrive" / folder_title
        if candidate.exists():
            return candidate / "ComfyUI_Master"
        # Search recursively in MyDrive (top 2 levels)
        for p in (DRIVE_MOUNT_POINT / "MyDrive").glob(f"**/{folder_title}"):
            if p.is_dir():
                return p / "ComfyUI_Master"
    except Exception:
        pass

    # Direct search for any folder matching common names or direct shortcut
    my_drive = DRIVE_MOUNT_POINT / "MyDrive"
    for item in my_drive.iterdir():
        if item.is_dir() and "comfy" in item.name.lower():
            return item

    return my_drive / "ComfyUI_Master"

DRIVE_MASTER_DIR = resolve_target_drive_dir()
CACHE_DIR = DRIVE_MASTER_DIR / "cache"
CACHE_FILE = CACHE_DIR / "comfy_env_cache.tar.gz"

def get_required_drive_dirs(base_dir: Path):
    return [
        base_dir / "models" / "checkpoints",
        base_dir / "models" / "loras",
        base_dir / "models" / "vae",
        base_dir / "models" / "controlnet",
        base_dir / "models" / "embeddings",
        base_dir / "models" / "clip",
        base_dir / "models" / "unet",
        base_dir / "models" / "diffusion_models",
        base_dir / "outputs",
        base_dir / "workflows",
        base_dir / "config",
        base_dir / "cache",
    ]


def run_cmd(cmd, env=None, check=True, cwd=None):
    """Execute shell command with real-time output and error checking."""
    print(f"[RUN] {cmd}")
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=cwd
    )
    output_lines = []
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        output_lines.append(line)
    process.stdout.close()
    return_code = process.wait()
    if check and return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd, output="".join(output_lines))
    return return_code, "".join(output_lines)

def check_gpu():
    """Verify GPU availability and identify CUDA version."""
    print("\n=== [1/6] GPU & CUDA Verification ===")
    try:
        ret, out = run_cmd("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False)
        if ret != 0:
            print("WARNING: nvidia-smi failed. GPU may not be enabled in Colab runtime settings.")
            return "cpu", "0.0"
        gpu_info = out.strip()
        print(f"Detected GPU: {gpu_info}")

        # Check CUDA version from nvidia-smi
        ret, out = run_cmd("nvidia-smi | grep -o 'CUDA Version: [0-9]*\\.[0-9]*'", check=False)
        cuda_version = out.replace("CUDA Version:", "").strip() if ret == 0 else "12.4"
        print(f"CUDA System Version: {cuda_version}")
        return gpu_info, cuda_version
    except Exception as e:
        print(f"Error checking GPU: {e}")
        return "unknown", "12.4"

def setup_google_drive():
    """Mount Google Drive and create persistent directory hierarchy."""
    global DRIVE_MASTER_DIR, CACHE_DIR, CACHE_FILE
    print("\n=== [2/6] Google Drive Setup & Directory Hierarchy ===")
    if not os.path.exists("/content/drive/MyDrive"):
        try:
            from google.colab import drive
            print("Mounting Google Drive to /content/drive ...")
            drive.mount(str(DRIVE_MOUNT_POINT))
        except ImportError:
            print("Not running in Google Colab environment. Creating local fallback directory...")

    DRIVE_MASTER_DIR = resolve_target_drive_dir()
    CACHE_DIR = DRIVE_MASTER_DIR / "cache"
    CACHE_FILE = CACHE_DIR / "comfy_env_cache.tar.gz"
    print(f"Using Google Drive Master Path: {DRIVE_MASTER_DIR}")

    for directory in get_required_drive_dirs(DRIVE_MASTER_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] Directory ready: {directory}")

    # Generate extra_model_paths.yaml if not present or update it
    config_yaml_path = DRIVE_MASTER_DIR / "config" / "extra_model_paths.yaml"
    yaml_content = f"""comfyui:
    base_path: {DRIVE_MASTER_DIR}
    checkpoints: models/checkpoints
    loras: models/loras
    vae: models/vae
    controlnet: models/controlnet
    embeddings: models/embeddings
    clip: models/clip
    unet: models/unet
    diffusion_models: models/diffusion_models
"""
    if not config_yaml_path.exists() or "diffusion_models" not in config_yaml_path.read_text():
        config_yaml_path.write_text(yaml_content)
        print(f"  [OK] Updated extra_model_paths.yaml with FLUX paths at {config_yaml_path}")

def restore_cache():
    """Check and unpack pre-built environment cache from Drive."""
    print("\n=== [3/6] Environment Cache Check ===")
    if CACHE_FILE.exists():
        print(f"Found existing cache at {CACHE_FILE} ({CACHE_FILE.stat().st_size / (1024*1024):.1f} MB)")
        print("Extracting cache to /content (using parallel decompression)...")
        start_time = time.time()
        # Ensure pigz is installed for ultra-fast parallel decompression
        run_cmd("command -v pigz >/dev/null 2>&1 || sudo apt-get install -y -qq pigz", check=False)
        tar_cmd = f"tar -I pigz -xf '{CACHE_FILE}' -C {LOCAL_WORK_DIR}"
        ret, out = run_cmd(tar_cmd, check=False)
        if ret == 0 and (MINICONDA_DIR / "envs" / CONDA_ENV_NAME).exists() and COMFYUI_DIR.exists():
            elapsed = time.time() - start_time
            print(f"[SUCCESS] Environment restored in {elapsed:.1f}s from cache!")
            return True
        else:
            print("[WARN] Cache extraction failed or was incomplete. Proceeding with fresh build...")
    else:
        print("No cache found. Performing fresh build...")
    return False

def install_isolated_conda():
    """Install Miniconda to isolate Python from Colab host upgrades."""
    print("\n=== [4/6] Isolated Python (Miniconda) Installation ===")
    if not MINICONDA_DIR.exists():
        installer_path = LOCAL_WORK_DIR / "miniconda.sh"
        # Download Miniconda with Python 3.10
        installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh"
        print(f"Downloading Miniconda from {installer_url} ...")
        run_cmd(f"wget -q --show-progress {installer_url} -O {installer_path}")
        run_cmd(f"bash {installer_path} -b -u -p {MINICONDA_DIR}")
        if installer_path.exists():
            installer_path.unlink()
        print(f"Miniconda successfully installed at {MINICONDA_DIR}")

    conda_bin = MINICONDA_DIR / "bin" / "conda"
    env_dir = MINICONDA_DIR / "envs" / CONDA_ENV_NAME
    if not env_dir.exists():
        print(f"Creating isolated conda environment: {CONDA_ENV_NAME} (Python 3.10) ...")
        run_cmd(f"{conda_bin} create -y -n {CONDA_ENV_NAME} python=3.10")

def patch_comfy_kitchen():
    """Fix PyTorch infer_schema compatibility bug with list[int] in comfy-kitchen."""
    print("Checking and patching comfy-kitchen for PyTorch infer_schema compatibility...")
    env_pip = MINICONDA_DIR / "envs" / CONDA_ENV_NAME / "bin" / "pip"
    run_cmd(f"{env_pip} install -U --no-deps comfy-kitchen", check=False)

    site_packages = MINICONDA_DIR / "envs" / CONDA_ENV_NAME / "lib" / "python3.10" / "site-packages"
    kitchen_dir = site_packages / "comfy_kitchen"
    if kitchen_dir.exists():
        for py_file in kitchen_dir.glob("**/*.py"):
            try:
                text = py_file.read_text(encoding="utf-8")
                if "list[int]" in text or "list[bool]" in text or "list[float]" in text:
                    patched = "import typing\n" + text
                    patched = patched.replace("list[int]", "typing.List[int]")
                    patched = patched.replace("list[bool]", "typing.List[bool]")
                    patched = patched.replace("list[float]", "typing.List[float]")
                    py_file.write_text(patched, encoding="utf-8")
                    print(f"  [PATCHED] {py_file.name}")
            except Exception as e:
                print(f"  [WARN] Failed to patch {py_file.name}: {e}")

def install_packages_with_retry(cuda_version):
    """Install PyTorch, xformers, and ComfyUI with autonomous error resolution."""
    print("\n=== [5/6] Library Installation & Dependencies ===")
    env_python = MINICONDA_DIR / "envs" / CONDA_ENV_NAME / "bin" / "python"
    env_pip = MINICONDA_DIR / "envs" / CONDA_ENV_NAME / "bin" / "pip"

    # Select appropriate PyTorch CUDA wheel
    # Colab T4/V100/A100/L4 supports cu121 or cu124
    cuda_tag = "cu124" if "12.4" in cuda_version or "12.5" in cuda_version or "12.6" in cuda_version else "cu121"
    torch_install_cmd = (
        f"{env_pip} install --no-cache-dir torch torchvision torchaudio "
        f"--index-url https://download.pytorch.org/whl/{cuda_tag}"
    )

    print(f"Installing PyTorch for CUDA {cuda_tag}...")
    ret, out = run_cmd(torch_install_cmd, check=False)
    if ret != 0:
        print("[RETRY] cu124 wheel failed, falling back to cu121 wheel...")
        fallback_cmd = (
            f"{env_pip} install --no-cache-dir torch torchvision torchaudio "
            f"--index-url https://download.pytorch.org/whl/cu121"
        )
        run_cmd(fallback_cmd)

    # Install xformers
    print("Installing xformers and acceleration packages...")
    ret, out = run_cmd(f"{env_pip} install --no-cache-dir xformers triton", check=False)
    if ret != 0:
        print("[WARN] Pre-built xformers wheel had issues, continuing with standard torch SDPA...")

    # Clone or update ComfyUI
    if not COMFYUI_DIR.exists():
        print("Cloning ComfyUI repository...")
        run_cmd(f"git clone https://github.com/comfyanonymous/ComfyUI.git {COMFYUI_DIR}")
    else:
        print("ComfyUI repository already exists. Updating...")
        run_cmd(f"git -C {COMFYUI_DIR} pull", check=False)

    # Install ComfyUI requirements
    print("Installing ComfyUI requirements...")
    req_file = COMFYUI_DIR / "requirements.txt"
    run_cmd(f"{env_pip} install -r {req_file}")

    # Install additional common utilities for headless/API operation
    print("Installing API utilities (aiohttp, websocket-client, requests)...")
    run_cmd(f"{env_pip} install aiohttp websocket-client requests pillow")

    # Autonomous Fix: Patch comfy-kitchen for PyTorch infer_schema compatibility (list[int] -> typing.List[int])
    patch_comfy_kitchen()

    # Link extra_model_paths.yaml into ComfyUI
    config_target = COMFYUI_DIR / "extra_model_paths.yaml"
    config_source = DRIVE_MASTER_DIR / "config" / "extra_model_paths.yaml"
    if not config_target.exists() and config_source.exists():
        os.symlink(config_source, config_target)
        print(f"  [OK] Symlinked extra_model_paths.yaml into {COMFYUI_DIR}")

def save_cache():
    """Archive isolated environment and ComfyUI to Drive for instant boots."""
    print("\n=== [6/6] Archiving Environment to Google Drive Cache ===")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temp_cache = LOCAL_WORK_DIR / "temp_cache.tar.gz"

    print("Archiving /content/miniconda and /content/ComfyUI ... (this takes ~1-2 min on first run)")
    start_time = time.time()
    run_cmd("command -v pigz >/dev/null 2>&1 || sudo apt-get install -y -qq pigz", check=False)
    
    # Exclude git objects or unnecessary caches to keep size compact
    archive_cmd = (
        f"tar -I pigz -cf '{temp_cache}' "
        f"--exclude='miniconda/pkgs' "
        f"--exclude='ComfyUI/.git' "
        f"-C {LOCAL_WORK_DIR} miniconda ComfyUI"
    )
    ret, out = run_cmd(archive_cmd, check=False)
    if ret == 0 and temp_cache.exists():
        shutil.move(str(temp_cache), str(CACHE_FILE))
        elapsed = time.time() - start_time
        size_mb = CACHE_FILE.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] Cache saved to {CACHE_FILE} ({size_mb:.1f} MB) in {elapsed:.1f}s!")
    else:
        print("[WARN] Failed to create environment cache archive.")

def main(force_rebuild=False):
    print("=================================================================")
    print("   ComfyUI Master Setup: Phase 1 Environment Initialization     ")
    print("=================================================================")
    start_total = time.time()

    gpu_info, cuda_version = check_gpu()
    setup_google_drive()

    cache_restored = False
    if not force_rebuild:
        cache_restored = restore_cache()

    if not cache_restored:
        install_isolated_conda()
        install_packages_with_retry(cuda_version)
        save_cache()
    else:
        patch_comfy_kitchen()

    # Re-verify symlink to Drive extra_model_paths.yaml
    config_target = COMFYUI_DIR / "extra_model_paths.yaml"
    config_source = DRIVE_MASTER_DIR / "config" / "extra_model_paths.yaml"
    if not config_target.exists() and config_source.exists():
        os.symlink(config_source, config_target)

    elapsed_total = time.time() - start_total
    print("\n=================================================================")
    print(f" [COMPLETE] Setup finished successfully in {elapsed_total:.1f}s!")
    print(f" - ComfyUI Directory: {COMFYUI_DIR}")
    print(f" - Conda Python: {MINICONDA_DIR / 'envs' / CONDA_ENV_NAME / 'bin' / 'python'}")
    print(f" - Persistent Models: {DRIVE_MASTER_DIR / 'models'}")
    print(f" - Outputs Dir: {DRIVE_MASTER_DIR / 'outputs'}")
    print("=================================================================")

if __name__ == "__main__":
    force = "--force-rebuild" in sys.argv
    main(force_rebuild=force)
