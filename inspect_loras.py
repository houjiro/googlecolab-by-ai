"""
LoRA Inspector & SDXL Compatibility Analyzer
Scans LoRA safetensors files and inspects metadata/tensor keys to identify base model (SDXL, SD 1.5, SD 2.1, Flux).
"""

import os
import sys
import json
import struct
from pathlib import Path
from typing import Dict, Any, Optional

def inspect_safetensors_header(file_path: Path) -> Dict[str, Any]:
    """Read the JSON header of a safetensors file without loading tensor data."""
    try:
        with open(file_path, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                return {"error": "Invalid file format: too small"}
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            if header_size > 100 * 1024 * 1024:  # Safety check
                return {"error": "Header too large"}
            header_json_bytes = f.read(header_size)
            header = json.loads(header_json_bytes.decode("utf-8"))
            return header
    except Exception as e:
        return {"error": str(e)}

def classify_lora(file_path: Path) -> Dict[str, Any]:
    """
    Classify a LoRA safetensors file into SDXL, SD 1.5, SD 2.x, or Flux.
    Checks:
      1. __metadata__ (ss_base_model_version, modelspec, etc.)
      2. Key architecture (lora_te1/lora_te2 for SDXL dual text encoder, etc.)
    """
    header = inspect_safetensors_header(file_path)
    if "error" in header:
        return {"file": file_path.name, "compatible_with_sdxl": False, "reason": header["error"]}

    metadata = header.get("__metadata__", {})
    keys = [k for k in header.keys() if k != "__metadata__"]

    base_model_version = metadata.get("ss_base_model_version", "").lower()
    base_model = metadata.get("ss_base_model", "").lower()
    network_module = metadata.get("ss_network_module", "").lower()

    # Direct metadata checks
    if "sdxl" in base_model_version or "sdxl" in base_model:
        return {
            "file": file_path.name,
            "compatible_with_sdxl": True,
            "architecture": "SDXL",
            "details": f"Explicit metadata: {base_model_version or base_model}"
        }

    if "flux" in base_model_version or "flux" in base_model:
        return {
            "file": file_path.name,
            "compatible_with_sdxl": False,
            "architecture": "Flux",
            "details": "Flux architecture"
        }

    if "v1" in base_model_version or "sd1" in base_model_version or "sd 1.5" in base_model:
        return {
            "file": file_path.name,
            "compatible_with_sdxl": False,
            "architecture": "SD 1.5",
            "details": f"Explicit metadata: {base_model_version or base_model}"
        }

    # Inspect key patterns
    has_te1 = any("lora_te1_" in k or "conditioner.embedders.0" in k for k in keys)
    has_te2 = any("lora_te2_" in k or "conditioner.embedders.1" in k for k in keys)
    has_sdxl_unet = any("input_blocks.4.1" in k or "label_emb" in k or "emb_layers" in k for k in keys)

    if (has_te1 and has_te2) or has_sdxl_unet:
        return {
            "file": file_path.name,
            "compatible_with_sdxl": True,
            "architecture": "SDXL",
            "details": "Detected SDXL dual text encoder / UNet keys"
        }

    # SD 1.5 pattern
    has_sd15_unet = any("middle_block.1.transformer_blocks" in k or "input_blocks.1.1" in k for k in keys)
    if has_sd15_unet and not (has_te1 and has_te2):
        return {
            "file": file_path.name,
            "compatible_with_sdxl": False,
            "architecture": "SD 1.5",
            "details": "Detected SD 1.5 single text encoder / UNet structure"
        }

    return {
        "file": file_path.name,
        "compatible_with_sdxl": False,
        "architecture": "Unknown",
        "details": "Could not determine architecture definitively"
    }

def scan_folder(folder_path: str):
    """Scan all safetensors files in target folder and report SDXL compatibility."""
    path = Path(folder_path)
    if not path.exists():
        print(f"[ERROR] Directory not found: {path}")
        return

    files = list(path.glob("**/*.safetensors"))
    if not files:
        print(f"No .safetensors files found in {path}")
        return

    print(f"\n=======================================================")
    print(f"  Scanning {len(files)} LoRA file(s) for SDXL Compatibility")
    print(f"=======================================================\n")

    sdxl_loras = []
    other_loras = []

    for f in sorted(files):
        res = classify_lora(f)
        if res.get("compatible_with_sdxl"):
            sdxl_loras.append(res)
        else:
            other_loras.append(res)

    print("🟢 [SDXL Compatible LoRAs]")
    if sdxl_loras:
        for item in sdxl_loras:
            print(f"  ✓ {item['file']} ({item['details']})")
    else:
        print("  (None found)")

    print("\n⚪ [Other / Incompatible with SDXL]")
    if other_loras:
        for item in other_loras:
            print(f"  ✗ {item['file']} [{item.get('architecture', 'Unknown')}] - {item.get('details', '')}")
    else:
        print("  (None)")

    print(f"\nSummary: {len(sdxl_loras)} SDXL LoRAs / {len(files)} total files.")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive"
    scan_folder(target)
