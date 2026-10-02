"""
ComfyUI Master Pipeline Tools for Google Colab Pro
Phase 2 & 3: Token Management and Autonomous API Wrappers
"""

import os
import sys
import json
import time
import uuid
import urllib.request
import urllib.parse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

# Colab / IPython Display
try:
    from IPython.display import Image as IPImage, display, HTML, clear_output
except ImportError:
    IPImage = None
    display = print

# Base Directories
TARGET_DRIVE_FOLDER_ID = "1CVwjVWJAR7PDiFbs4IPCoKzE3KYzDk2n"
DRIVE_MOUNT_POINT = Path("/content/drive")
LOCAL_COMFY_DIR = Path("/content/ComfyUI")
CONDA_PYTHON = Path("/content/miniconda/envs/comfy_env/bin/python")
CONDA_PIP = Path("/content/miniconda/envs/comfy_env/bin/pip")
COMFY_API_URL = "http://127.0.0.1:8188"

def resolve_target_drive_dir() -> Path:
    """Resolve target directory from Google Drive."""
    my_drive = DRIVE_MOUNT_POINT / "MyDrive"
    if not my_drive.exists():
        return my_drive / "ComfyUI_Master"

    try:
        from setup_comfyui import resolve_target_drive_dir as resolver
        return resolver()
    except Exception:
        pass

    for item in my_drive.iterdir():
        if item.is_dir() and "comfy" in item.name.lower():
            return item
    return my_drive / "ComfyUI_Master"

DRIVE_MASTER_DIR = resolve_target_drive_dir()
TOKENS_CONFIG_PATH = DRIVE_MASTER_DIR / "config" / "tokens.json"

# =====================================================================
# Phase 2: Token & Credential Management
# =====================================================================

def get_token(key: str = "hf_token") -> Optional[str]:
    """
    Retrieve token securely from Drive config/tokens.json or environment.
    Strictly avoids hardcoded tokens.
    """
    # 1. Check Drive tokens.json
    if TOKENS_CONFIG_PATH.exists():
        try:
            with open(TOKENS_CONFIG_PATH, "r", encoding="utf-8") as f:
                tokens = json.load(f)
                token = tokens.get(key)
                if token and token.strip() and not token.startswith("PASTE_YOUR_"):
                    return token.strip()
        except Exception as e:
            print(f"[WARN] Failed to read {TOKENS_CONFIG_PATH}: {e}")

    # 2. Check Colab Secrets / Environment variables
    env_token = os.environ.get(key.upper()) or os.environ.get(key)
    if env_token and env_token.strip():
        return env_token.strip()

    return None

def require_hf_token() -> str:
    """
    Ensure Hugging Face token is present.
    If missing, prompts the user to provide it and creates a template in Drive.
    """
    token = get_token("hf_token")
    if token:
        return token

    # Guide user to set token
    TOKENS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not TOKENS_CONFIG_PATH.exists():
        sample_json = {
            "hf_token": "PASTE_YOUR_HUGGINGFACE_TOKEN_HERE",
            "civitai_api_key": ""
        }
        with open(TOKENS_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(sample_json, f, indent=4)

    err_msg = (
        f"\n[AUTHENTICATION REQUIRED]\n"
        f"Hugging Face token is required for this operation.\n"
        f"Please write your token into:\n"
        f"  '{TOKENS_CONFIG_PATH}'\n"
        f"Format: {{\"hf_token\": \"hf_xxxxxxxxxxxx\"}}\n"
        f"Or set the HF_TOKEN environment variable.\n"
    )
    print(err_msg)
    raise PermissionError("Missing Hugging Face token. Please configure tokens.json on Google Drive.")

# =====================================================================
# Phase 3: Autonomous API Wrapper Functions
# =====================================================================

def run_pip(args_str: str):
    """Run pip inside the isolated conda environment."""
    cmd = f"{CONDA_PIP} {args_str}"
    print(f"[PIP] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def install_model(url_or_hf_id: str, target_dir: Optional[str] = None, filename: Optional[str] = None) -> Path:
    """
    1. install_model(url_or_hf_id, target_dir):
    Downloads model checkpoints, LoRAs, or VAEs into Google Drive storage.
    Supports direct URLs, Hugging Face repos/files, with aria2c acceleration and token auth.
    """
    # Auto-detect target directory if not specified
    if not target_dir:
        lower_src = url_or_hf_id.lower()
        if "lora" in lower_src:
            target_dir = "models/loras"
        elif "vae" in lower_src:
            target_dir = "models/vae"
        elif "controlnet" in lower_src:
            target_dir = "models/controlnet"
        else:
            target_dir = "models/checkpoints"

    dest_dir = DRIVE_MASTER_DIR / target_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename
    if not filename:
        if "/" in url_or_hf_id:
            filename = url_or_hf_id.split("/")[-1].split("?")[0]
        else:
            filename = f"{url_or_hf_id}.safetensors"

    dest_file = dest_dir / filename
    if dest_file.exists() and dest_file.stat().st_size > 1024 * 1024:
        print(f"[SKIP] Model already exists: {dest_file} ({dest_file.stat().st_size / (1024*1024):.1f} MB)")
        return dest_file

    print(f"\n[DOWNLOAD] Target: {dest_file}")
    print(f"Source: {url_or_hf_id}")

    # Case A: Hugging Face ID (e.g., username/repo or huggingface.co URL)
    is_hf = "huggingface.co" in url_or_hf_id or (url_or_hf_id.count("/") == 1 and not url_or_hf_id.startswith("http"))
    headers = []
    hf_token = get_token("hf_token")
    if is_hf and hf_token:
        headers.append(f"Authorization: Bearer {hf_token}")

    # If it is an HF repo identifier (e.g. runwayml/stable-diffusion-v1-5 and single file)
    download_url = url_or_hf_id
    if is_hf and not url_or_hf_id.startswith("http"):
        # Default to huggingface resolve URL if a specific safetensors file is provided
        if filename.endswith(".safetensors") or filename.endswith(".ckpt"):
            download_url = f"https://huggingface.co/{url_or_hf_id}/resolve/main/{filename}"
        else:
            # Fallback to downloading entire repo via huggingface_hub in conda env
            print("Downloading from Hugging Face Hub using isolated env...")
            script = f"""
from huggingface_hub import snapshot_download
snapshot_download(repo_id='{url_or_hf_id}', local_dir='{dest_dir}', token='{hf_token or ""}')
"""
            subprocess.run([str(CONDA_PYTHON), "-c", script], check=True)
            return dest_dir

    # Use aria2c for maximum download throughput on Colab
    subprocess.run("command -v aria2c >/dev/null 2>&1 || sudo apt-get install -y -qq aria2c", shell=True, check=False)
    header_args = " ".join([f'--header="{h}"' for h in headers])
    aria_cmd = (
        f"aria2c -c -x 16 -s 16 -k 1M {header_args} "
        f"-d '{dest_dir}' -o '{filename}' '{download_url}'"
    )

    ret = subprocess.run(aria_cmd, shell=True)
    if ret.returncode != 0 or not dest_file.exists() or dest_file.stat().st_size == 0:
        print("[WARN] aria2c download failed or file empty. Falling back to curl...")
        curl_headers = " ".join([f'-H "{h}"' for h in headers])
        curl_cmd = f"curl -L -C - {curl_headers} -o '{dest_file}' '{download_url}'"
        subprocess.run(curl_cmd, shell=True, check=True)

    print(f"[SUCCESS] Model saved to {dest_file} ({dest_file.stat().st_size / (1024*1024):.1f} MB)")
    return dest_file

def install_custom_node(git_url: str) -> Path:
    """
    2. install_custom_node(git_url):
    Clones custom node repository into ComfyUI/custom_nodes and installs dependencies into isolated env.
    """
    custom_nodes_dir = LOCAL_COMFY_DIR / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)

    repo_name = git_url.rstrip("/").split("/")[-1].replace(".git", "")
    target_repo_dir = custom_nodes_dir / repo_name

    if target_repo_dir.exists():
        print(f"[UPDATE] Custom node already exists at {target_repo_dir}. Pulling latest...")
        subprocess.run(f"git -C '{target_repo_dir}' pull", shell=True, check=False)
    else:
        print(f"[CLONE] Installing custom node from {git_url} ...")
        subprocess.run(f"git clone '{git_url}' '{target_repo_dir}'", shell=True, check=True)

    # Check and install requirements.txt
    req_file = target_repo_dir / "requirements.txt"
    if req_file.exists():
        print(f"Installing dependencies for {repo_name}...")
        run_pip(f"install -r '{req_file}'")

    print(f"[SUCCESS] Custom node installed: {repo_name}")
    print("[NOTE] If ComfyUI is already running, a restart may be required for nodes to load.")
    return target_repo_dir

def generate_media(prompt_json: Dict[str, Any], api_url: str = COMFY_API_URL, timeout_sec: int = 600) -> List[Path]:
    """
    3. generate_media(prompt_json):
    Submits a prompt workflow JSON to ComfyUI API (http://127.0.0.1:8188) and tracks completion.
    Returns list of paths to generated outputs.
    """
    client_id = str(uuid.uuid4())
    payload = json.dumps({"prompt": prompt_json, "client_id": client_id}).encode("utf-8")

    # Submit prompt
    req = urllib.request.Request(
        f"{api_url}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            prompt_id = resp_data.get("prompt_id")
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to connect to ComfyUI API at {api_url}. Is server running? Error: {e}")

    print(f"[QUEUED] Job submitted. Prompt ID: {prompt_id}")

    # Track job execution via polling /history
    start_time = time.time()
    last_node = ""
    while time.time() - start_time < timeout_sec:
        time.sleep(1.0)
        try:
            with urllib.request.urlopen(f"{api_url}/history/{prompt_id}") as resp:
                history_data = json.loads(resp.read().decode("utf-8"))
                if prompt_id in history_data:
                    prompt_history = history_data[prompt_id]
                    # Check status
                    status = prompt_history.get("status", {})
                    if status.get("status_str") == "error":
                        raise RuntimeError(f"ComfyUI execution error: {status.get('messages')}")

                    # Collect output images
                    outputs = prompt_history.get("outputs", {})
                    output_files: List[Path] = []
                    for node_id, node_output in outputs.items():
                        images = node_output.get("images", [])
                        for img in images:
                            fname = img.get("filename")
                            subfolder = img.get("subfolder", "")
                            # Check in Drive outputs dir first, then ComfyUI/output
                            drive_out = DRIVE_MASTER_DIR / "outputs" / subfolder / fname
                            local_out = LOCAL_COMFY_DIR / "output" / subfolder / fname
                            if drive_out.exists():
                                output_files.append(drive_out)
                            elif local_out.exists():
                                output_files.append(local_out)
                            else:
                                output_files.append(drive_out) # Target expected path

                    elapsed = time.time() - start_time
                    print(f"\n[DONE] Generation finished in {elapsed:.1f}s. Produced {len(output_files)} file(s).")
                    return output_files
        except Exception as e:
            if "ComfyUI execution error" in str(e):
                raise
            # Continue polling if history not ready yet
            pass

    raise TimeoutError(f"Generation timed out after {timeout_sec} seconds.")

def preview_image(image_path: Path):
    """
    4. preview_image(image_path):
    Displays image or video directly in the notebook output using IPython.display.
    Does not use ngrok or external tunnels.
    """
    path = Path(image_path)
    if not path.exists():
        print(f"[ERROR] File does not exist for preview: {path}")
        return

    ext = path.suffix.lower()
    print(f"\n[PREVIEW] Displaying: {path.name}")
    if ext in [".png", ".jpg", ".jpeg", ".webp"]:
        if IPImage:
            display(IPImage(filename=str(path)))
        else:
            print(f"[INFO] Image saved at {path}")
    elif ext in [".mp4", ".webm"]:
        video_html = f"""
        <video width="640" height="480" controls autoplay loop>
            <source src="file://{path.resolve()}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """
        display(HTML(video_html))
    else:
        print(f"[INFO] File generated at: {path}")

# =====================================================================
# Standard Workflow Helper (Phase 4 Base)
# =====================================================================

def create_default_txt2img_workflow(
    ckpt_name: str,
    positive_prompt: str,
    negative_prompt: str = "ugly, blurry, low quality, artifacts, watermark",
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg: float = 7.0,
    sampler_name: str = "euler_ancestral",
    scheduler: str = "normal",
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generates standard ComfyUI API prompt JSON for text-to-image (SD 1.5)."""
    if seed is None:
        seed = int(time.time() * 1000) % 10000000000

    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg,
                "denoise": 1,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "seed": seed,
                "steps": steps
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": ckpt_name
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "batch_size": 1,
                "height": height,
                "width": width
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": positive_prompt
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": negative_prompt
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "ComfyUI_Master",
                "images": ["8", 0]
            }
        }
    }
    return workflow

def create_sdxl_txt2img_workflow(
    ckpt_name: str,
    positive_prompt: str,
    negative_prompt: str = "ugly, blurry, low quality, artifacts, distorted, bad anatomy, watermark",
    width: int = 1024,
    height: int = 1024,
    steps: int = 30,
    cfg: float = 7.0,
    sampler_name: str = "euler_ancestral",
    scheduler: str = "normal",
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generates standard ComfyUI API prompt JSON optimized for SDXL (1024x1024)."""
    return create_default_txt2img_workflow(
        ckpt_name=ckpt_name,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        seed=seed
    )

def parse_multi_loras(lora_input: Optional[str], default_strength: float = 0.85) -> List[tuple]:
    """
    Parses comma-separated LoRA string into list of (filename, strength).
    Supports format: 'lora1:0.8, lora2:0.5' or 'lora1, lora2'
    """
    if not lora_input or not str(lora_input).strip():
        return []
    
    results = []
    items = [x.strip() for x in str(lora_input).split(",") if x.strip()]
    for item in items:
        if ":" in item:
            parts = item.split(":")
            name = parts[0].strip()
            try:
                strength = float(parts[1].strip())
            except ValueError:
                strength = default_strength
        else:
            name = item.strip()
            strength = default_strength
        
        if not name.endswith(".safetensors"):
            name = f"{name}.safetensors"
        results.append((name, strength))
    return results

def list_drive_models():
    """Outputs clear list of all Checkpoints, FLUX models, LoRAs, and Upscalers on Google Drive."""
    from inspect_loras import classify_lora
    print("=========================================================================================")
    print(f" 📂 Google Drive 読み込み済みモデル一覧 (場所: {DRIVE_MASTER_DIR})")
    print("=========================================================================================")
    
    # 1. Checkpoints (SDXL)
    ckpt_dir = DRIVE_MASTER_DIR / "models" / "checkpoints"
    ckpts = sorted(list(ckpt_dir.rglob("*.safetensors")) + list(ckpt_dir.rglob("*.ckpt"))) if ckpt_dir.exists() else []
    print(f"\n📦 Checkpoints (SDXL / models/checkpoints): {len(ckpts)} 件")
    if ckpts:
        for f in ckpts:
            size_gb = f.stat().st_size / (1024*1024*1024)
            print(f"  • {f.name:<40} ({size_gb:5.2f} GB)")
    else:
        print("  (なし)")

    # 2. Diffusion Models (FLUX)
    unet_dir = DRIVE_MASTER_DIR / "models" / "unet"
    unets = sorted(list(unet_dir.rglob("*.safetensors"))) if unet_dir.exists() else []
    print(f"\n⚡ Diffusion Models (FLUX本体 / models/unet): {len(unets)} 件")
    if unets:
        for f in unets:
            size_gb = f.stat().st_size / (1024*1024*1024)
            print(f"  • {f.name:<40} ({size_gb:5.2f} GB)")
    else:
        print("  (なし)")

    # 3. LoRAs
    lora_dir = DRIVE_MASTER_DIR / "models" / "loras"
    loras = sorted(list(lora_dir.rglob("*.safetensors"))) if lora_dir.exists() else []
    print(f"\n🎀 LoRAs (models/loras): {len(loras)} 件")
    if loras:
        for f in loras:
            size_mb = f.stat().st_size / (1024*1024)
            try:
                diag = classify_lora(f)
                badge = "🟢 [SDXL対応]" if diag.get("compatible_with_sdxl") else f"🔵 [{diag.get('architecture', 'FLUX/Other')}]"
            except Exception:
                badge = ""
            print(f"  • {f.name:<38} ({size_mb:6.1f} MB) {badge}")
    else:
        print("  (なし)")

    # 4. Upscalers
    up_dir = DRIVE_MASTER_DIR / "models" / "upscale_models"
    ups = sorted(list(up_dir.rglob("*.pth")) + list(up_dir.rglob("*.safetensors"))) if up_dir.exists() else []
    print(f"\n🔍 Upscale Models (models/upscale_models): {len(ups)} 件")
    if ups:
        for f in ups:
            size_mb = f.stat().st_size / (1024*1024)
            print(f"  • {f.name:<40} ({size_mb:5.1f} MB)")
    else:
        print("  (なし)")
    print("=========================================================================================\n")

def create_sdxl_lora_workflow(
    ckpt_name: str,
    lora_name: Optional[str] = None,
    positive_prompt: str = "",
    negative_prompt: str = "ugly, blurry, low quality, artifacts, distorted, bad anatomy, watermark",
    lora_strength: float = 0.85,
    width: int = 1344,
    height: int = 768,
    steps: int = 30,
    cfg: float = 7.0,
    sampler_name: str = "euler_ancestral",
    scheduler: str = "normal",
    batch_size: int = 1,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generates ComfyUI API prompt JSON for SDXL with Multi-LoRA support."""
    if seed is None:
        seed = int(time.time() * 1000) % 10000000000

    workflow = {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt_name}
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "batch_size": batch_size,
                "height": height,
                "width": width
            }
        }
    }

    # Handle Multi-LoRA Chaining
    model_source = ["4", 0]
    clip_source = ["4", 1]
    
    lora_list = parse_multi_loras(lora_name, lora_strength)
    node_id = 100
    for l_fname, l_str in lora_list:
        cur_id = str(node_id)
        workflow[cur_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": l_fname,
                "strength_model": l_str,
                "strength_clip": l_str,
                "model": model_source,
                "clip": clip_source
            }
        }
        model_source = [cur_id, 0]
        clip_source = [cur_id, 1]
        node_id += 1

    workflow["6"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": clip_source, "text": positive_prompt}
    }
    workflow["7"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": clip_source, "text": negative_prompt}
    }
    workflow["3"] = {
        "class_type": "KSampler",
        "inputs": {
            "cfg": cfg,
            "denoise": 1.0,
            "latent_image": ["5", 0],
            "model": model_source,
            "positive": ["6", 0],
            "negative": ["7", 0],
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "seed": seed,
            "steps": steps
        }
    }
    workflow["8"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
    }
    workflow["9"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ComfyUI_Master", "images": ["8", 0]}
    }
    return workflow

def create_flux_workflow(
    positive_prompt: str,
    lora_name: Optional[str] = None,
    lora_strength: float = 0.85,
    unet_name: str = "flux1-dev-fp8.safetensors",
    clip_name1: str = "t5xxl_fp8_e4m3fn.safetensors",
    clip_name2: str = "clip_l.safetensors",
    vae_name: str = "ae.safetensors",
    width: int = 1344,
    height: int = 768,
    steps: int = 25,
    guidance: float = 3.5,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    batch_size: int = 1,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generates ComfyUI API prompt JSON for FLUX.1 [dev] with Multi-LoRA support."""
    if seed is None:
        seed = int(time.time() * 1000) % 10000000000

    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"}
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {"clip_name1": clip_name1, "clip_name2": clip_name2, "type": "flux"}
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name}
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": batch_size, "height": height, "width": width}
        }
    }

    # Handle Multi-LoRA Chaining for FLUX
    model_source = ["1", 0]
    clip_source = ["2", 0]

    lora_list = parse_multi_loras(lora_name, lora_strength)
    node_id = 100
    for l_fname, l_str in lora_list:
        cur_id = str(node_id)
        workflow[cur_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": l_fname,
                "strength_model": l_str,
                "strength_clip": l_str,
                "model": model_source,
                "clip": clip_source
            }
        }
        model_source = [cur_id, 0]
        clip_source = [cur_id, 1]
        node_id += 1

    workflow["6"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": clip_source, "text": positive_prompt}
    }
    workflow["7"] = {
        "class_type": "FluxGuidance",
        "inputs": {"guidance": guidance, "conditioning": ["6", 0]}
    }
    workflow["8"] = {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 1.0,
            "denoise": 1.0,
            "latent_image": ["5", 0],
            "model": model_source,
            "positive": ["7", 0],
            "negative": ["6", 0],
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "seed": seed,
            "steps": steps
        }
    }
    workflow["9"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["8", 0], "vae": ["3", 0]}
    }
    workflow["11"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "FLUX_Master", "images": ["9", 0]}
    }
    return workflow

def run_generation_panel(
    engine: str,
    prompt: str,
    negative_prompt: str,
    aspect_ratio_str: str,
    batch_count: int = 1,
    concurrent_batch_size: int = 1,
    apply_lora: bool = False,
    lora_file: str = "",
    lora_strength: float = 0.85,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    upscale_2x: bool = False,
    steps: int = 25,
    guidance_or_cfg: float = 3.5,
    seed: int = -1
):
    """Unified runner for Colab Form UI inputs with 2x Upscaling and Batch controls."""
    # Parse Resolution
    res_map = {
        "16:9 (1344x768)": (1344, 768),
        "1:1 (1024x1024)": (1024, 1024),
        "9:16 (768x1344)": (768, 1344),
        "4:3 (1152x864)": (1152, 864),
        "3:4 (864x1152)": (864, 1152),
    }
    width, height = res_map.get(aspect_ratio_str, (1024, 1024))
    
    # Resolve LoRA (support single or comma-separated multi-LoRA)
    active_lora = None
    if apply_lora and lora_file.strip():
        active_lora = lora_file.strip()

    # Handle Upscaler Model check
    upscaler_file = "4x-UltraSharp.pth"
    if upscale_2x:
        upscale_dir = DRIVE_MASTER_DIR / "models" / "upscale_models"
        upscale_dir.mkdir(parents=True, exist_ok=True)
        target_pth = upscale_dir / upscaler_file
        if not target_pth.exists() or target_pth.stat().st_size < 1024 * 1024:
            print("Downloading 4x-UltraSharp upscaler model (~67MB)...")
            upscale_url = "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"
            try:
                install_model(upscale_url, target_dir="models/upscale_models", filename=upscaler_file)
            except Exception as e:
                print(f"[WARN] Failed to download upscaler: {e}")

    is_flux = "FLUX" in engine.upper()
    print(f"\n=======================================================")
    print(f" 🎨 生成開始: 【{engine}】 | 解像度: {width}x{height} (2x拡大: {'ON (2688x1536)' if upscale_2x else 'OFF'})")
    print(f" 🔢 生成枚数: {batch_count} 枚 (同時バッチ: {concurrent_batch_size})")
    print(f" ⚙️ Sampler: {sampler_name} | Scheduler: {scheduler} | Steps: {steps}")
    if active_lora:
        parsed_loras = parse_multi_loras(active_lora, lora_strength)
        lora_desc = ", ".join([f"{n} (強度:{s})" for n, s in parsed_loras])
        print(f" 🎀 適用LoRA: {lora_desc}")
    print(f"=======================================================")

    for i in range(1, batch_count + 1):
        cur_seed = int(time.time() * 1000 + i * 9973) % 10000000000 if seed == -1 else (seed + i - 1)
        print(f"\n▶ [{i}/{batch_count} 枚目] 生成中 (Seed: {cur_seed})...")
        
        if is_flux:
            wf = create_flux_workflow(
                positive_prompt=prompt,
                lora_name=active_lora,
                lora_strength=lora_strength,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance_or_cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                batch_size=concurrent_batch_size,
                seed=cur_seed
            )
            # Inject 2x Upscale nodes to FLUX workflow
            if upscale_2x:
                wf["20"] = {
                    "class_type": "UpscaleModelLoader",
                    "inputs": {"model_name": upscaler_file}
                }
                wf["21"] = {
                    "class_type": "ImageUpscaleWithModel",
                    "inputs": {"upscale_model": ["20", 0], "image": ["9", 0]}
                }
                wf["22"] = {
                    "class_type": "ImageScaleBy",
                    "inputs": {"image": ["21", 0], "upscale_method": "bicubic", "scale_by": 0.5}
                }
                wf["11"]["inputs"]["images"] = ["22", 0]
        else:
            if active_lora:
                wf = create_sdxl_lora_workflow(
                    ckpt_name="sd_xl_base_1.0.safetensors",
                    lora_name=active_lora,
                    positive_prompt=prompt,
                    negative_prompt=negative_prompt,
                    lora_strength=lora_strength,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg=guidance_or_cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    batch_size=concurrent_batch_size,
                    seed=cur_seed
                )
            else:
                wf = create_sdxl_txt2img_workflow(
                    ckpt_name="sd_xl_base_1.0.safetensors",
                    positive_prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg=guidance_or_cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    seed=cur_seed
                )
                wf["5"]["inputs"]["batch_size"] = concurrent_batch_size
            
            # Inject 2x Upscale nodes to SDXL workflow
            if upscale_2x:
                wf["20"] = {
                    "class_type": "UpscaleModelLoader",
                    "inputs": {"model_name": upscaler_file}
                }
                wf["21"] = {
                    "class_type": "ImageUpscaleWithModel",
                    "inputs": {"upscale_model": ["20", 0], "image": ["8", 0]}
                }
                wf["22"] = {
                    "class_type": "ImageScaleBy",
                    "inputs": {"image": ["21", 0], "upscale_method": "bicubic", "scale_by": 0.5}
                }
                wf["9"]["inputs"]["images"] = ["22", 0]
        
        outputs = generate_media(wf)
        for img in outputs:
            preview_image(img)
    
    print("\n🎉 すべての生成・処理が完了しました！")
