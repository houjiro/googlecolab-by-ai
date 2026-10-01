# ComfyUI Master Pipeline on Google Colab Pro

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/houjiro/googlecolab-by-ai/blob/main/ComfyUI_Colab_Master.ipynb)

Google Colab ProのリソースとGoogle Driveを連携させ、ComfyUIを用いた自律的な画像/動画生成パイプラインを運用するためのリポジトリです。

- **GitHub Repository**: [https://github.com/houjiro/googlecolab-by-ai](https://github.com/houjiro/googlecolab-by-ai)
- **Target Google Drive Folder ID**: `1CVwjVWJAR7PDiFbs4IPCoKzE3KYzDk2n`  
  ([Google Drive Folder Link](https://drive.google.com/drive/u/0/folders/1CVwjVWJAR7PDiFbs4IPCoKzE3KYzDk2n))

---

## クイックスタート

1. 上の **「Open in Colab」バッジ** をクリックして Google Colab でノートブックを開きます。
2. Colab のメニューで **「ランタイム」 > 「ランタイムのタイプを変更」** を開き、ハードウェアアクセラレータを **GPU (T4 / A100 / L4)** に設定します。
3. セルを上から順に実行します。
   - **セクション 0**: 最新スクリプトを自動同期
   - **セクション 1**: Google Driveマウントと仮想環境（Miniconda）のセットアップ
   - **セクション 2**: ComfyUI サーバーの起動
   - **セクション 3**: 自律APIツールの読み込み
   - **セクション 4**: SDXL Base 1.0 (約6.9GB) & VAE のダウンロード（Driveへ保存）
   - **セクション 5**: 1024×1024 高精細画像の生成＆ノートブック上プレビュー

---

## 構成ファイル

- **`ComfyUI_Colab_Master.ipynb`**:
  Colab上で一気通貫で実行できるメインノートブック。
- **`setup_comfyui.py`**:
  - Google Driveマウントと指定フォルダの自動解決
  - Miniconda (Python 3.10) 環境隔離とPyTorch / xformers / ComfyUI のインストール
  - Driveキャッシュによる2回目以降の高速起動（約10〜20秒）
  - `extra_model_paths.yaml` のDrive自動リンク
- **`pipeline_tools.py`**:
  - トークン安全管理（`tokens.json`）
  - `install_model` / `install_custom_node` / `generate_media` / `preview_image`
  - SD 1.5 & SDXL 用ワークフロー生成関数
- **`download_sdxl.py`**:
  - SDXL Base 1.0 モデルおよび FP16 修正版 VAE の高速ダウンロードスクリプト

---

## Google Drive フォルダ構造

指定されたGoogle Driveフォルダ内に、自動的に以下の階層が整備されます：

```text
ComfyUI_Master/
├── models/
│   ├── checkpoints/      # メインモデル (sd_xl_base_1.0.safetensors 等)
│   ├── loras/            # LoRAモデル
│   ├── vae/              # VAE (sdxl.vae.safetensors 等)
│   ├── controlnet/       # ControlNet
│   ├── embeddings/       # Textual Inversion
│   └── clip/             # CLIPモデル
├── outputs/              # 生成画像の自動保存先（永続化）
├── workflows/            # ワークフローJSON
├── config/
│   ├── extra_model_paths.yaml
│   └── tokens.json       # Hugging Face等のトークン設定
└── cache/
    └── comfy_env_cache.tar.gz  # 環境の高速復元用アーカイブ
```
