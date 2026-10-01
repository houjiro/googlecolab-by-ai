"""
Google Drive File Utilities for ComfyUI Colab Pipeline
- Exact-name file copy across Drive folders
- Batch remove ' のコピー' / 'Copy of ' from filenames
"""

import os
import shutil
import re
from pathlib import Path

def copy_without_renaming(src_dir: str, dst_dir: str, pattern: str = "*.safetensors"):
    """
    Copies files from src_dir to dst_dir preserving original exact filenames.
    No 'のコピー' or 'Copy of' added.
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    files = list(src_path.glob(pattern))
    print(f"[COPY] Found {len(files)} file(s) matching '{pattern}' in {src_path}")

    copied_count = 0
    for f in files:
        target = dst_path / f.name
        if target.exists() and target.stat().st_size == f.stat().st_size:
            print(f"  [SKIP] Already exists: {f.name}")
            continue
        print(f"  [COPYING] {f.name} ({f.stat().st_size / (1024*1024):.1f} MB) -> {dst_path.name}/")
        shutil.copy2(f, target)
        copied_count += 1

    print(f"\n[DONE] Successfully copied {copied_count} file(s) with exact original names!")

def clean_copy_suffixes(target_dir: str):
    """
    Removes ' のコピー', 'のコピー', ' - コピー', 'Copy of ' from all files in target_dir.
    """
    dir_path = Path(target_dir)
    if not dir_path.exists():
        print(f"[ERROR] Directory not found: {dir_path}")
        return

    files = [f for f in dir_path.iterdir() if f.is_file()]
    renamed_count = 0

    for f in files:
        orig_name = f.name
        new_name = orig_name
        # Japanese patterns
        new_name = re.sub(r'[\s\-_]*のコピー(?:\s*\(\d+\))?', '', new_name)
        new_name = re.sub(r'[\s\-_]*コピー(?:\s*\(\d+\))?', '', new_name)
        # English patterns
        new_name = re.sub(r'^Copy of\s*', '', new_name, flags=re.IGNORECASE)
        new_name = re.sub(r'[\s\-_]*copy(?:\s*\(\d+\))?', '', new_name, flags=re.IGNORECASE)

        if new_name != orig_name:
            target_path = dir_path / new_name
            if target_path.exists():
                print(f"  [SKIP] Destination name already exists: {new_name}")
                continue
            f.rename(target_path)
            print(f"  [RENAMED] '{orig_name}' -> '{new_name}'")
            renamed_count += 1

    print(f"\n[DONE] Cleaned filenames for {renamed_count} file(s) in {dir_path.name}/")

if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "clean"
    if action == "clean" and len(sys.argv) > 2:
        clean_copy_suffixes(sys.argv[2])
    elif action == "copy" and len(sys.argv) > 3:
        pattern = sys.argv[4] if len(sys.argv) > 4 else "*.safetensors"
        copy_without_renaming(sys.argv[2], sys.argv[3], pattern)
    else:
        print("Usage:")
        print("  python drive_utils.py clean <target_directory>")
        print("  python drive_utils.py copy <source_dir> <destination_dir> [pattern]")
