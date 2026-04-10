# Multimodal Drive Scene Understanding

This repository contains a notebook for visual question answering (VQA) on driving scenes from the nuScenes mini dataset using multimodal foundation models.

## Repository Contents

- `notebooks/Driving_VQA_v1.ipynb`: End-to-end notebook for loading nuScenes camera images and asking natural language questions about each scene.
- Colab notebook: [Open in Google Colab](https://colab.research.google.com/drive/1iLR6eBwKxxSYHYqR80B5xrW_zckrOIlo?usp=sharing)

## What the Notebook Covers

The notebook implements two VQA pipelines:

1. **BLIP-2 pipeline**
   - Model: `Salesforce/blip2-opt-2.7b`
   - Uses `Blip2Processor` and `Blip2ForConditionalGeneration`
   - Includes single-image interactive Q&A and multi-sample batch evaluation

2. **Qwen2-VL pipeline**
   - Model: `Qwen/Qwen2-VL-2B-Instruct`
   - Uses `Qwen2VLForConditionalGeneration` and `AutoProcessor`
   - Includes single-image test prompts and an interactive question widget

Both sections read `CAM_FRONT` images from nuScenes mini (`v1.0-mini`) and generate short answers to user questions.

## Environment and Dependencies

The notebook is designed for **Google Colab** with GPU (recommended: T4) and installs packages in cells, including:

- `transformers`
- `accelerate`
- `nuscenes-devkit`
- `qwen-vl-utils`
- `Pillow`
- `matplotlib`
- `ipywidgets`

## Dataset Setup

The notebook expects nuScenes data under a mounted Google Drive path:

- `NUSCENES_ROOT = /content/drive/MyDrive/nuscenes/nuscenes_dataset`

Expected folders include:

- `maps`
- `samples`
- `sweeps`
- `v1.0-mini`

Update `NUSCENES_ROOT` in the notebook if your dataset path is different.

## Quick Start

1. Open `notebooks/Driving_VQA_v1.ipynb` in Google Colab.
2. Enable GPU runtime (`Runtime -> Change runtime type -> T4 GPU`).
3. Run cells in order:
   - Install dependencies
   - Mount Google Drive
   - Verify nuScenes mini structure
   - Load model and processor
   - Run VQA on `CAM_FRONT` samples
4. Use interactive text widgets to ask scene questions (for example, traffic conditions, lane state, and nearby objects).

## Notes

- The notebook uses `float16` inference and auto-detects CUDA when available.
- Model downloads are large and may take a few minutes on first run.