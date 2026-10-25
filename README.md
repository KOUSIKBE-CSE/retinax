# RetinaX

RetinaX is a modular research prototype for five-class diabetic-retinopathy classification, explainability, federated learning, and lesion segmentation.

## Status and data requirements

The classification pipeline accepts an `ImageFolder` dataset with `Mild`, `Moderate`, `No_DR`, `Proliferate_DR`, and `Severe` folders. The supplied class-folder dataset does **not** provide pixel masks; it cannot train Attention U-Net. Configure a separate paired image/mask dataset before enabling segmentation.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m retinax.cli inspect-data --data-root /content/drive/MyDrive/gaussian_filtered_images
python -m retinax.cli train --data-root /content/drive/MyDrive/gaussian_filtered_images --epochs 5
```

Training saves the model and held-out test metrics:

```bash
python -m retinax.cli train \
  --data-root "/path/to/gaussian_filtered_images" \
  --model baseline \
  --epochs 15 \
  --batch-size 32 \
  --checkpoint checkpoints/retinax_baseline.pt
```

The checkpoint can be entered in the Streamlit sidebar. It contains model weights, class metadata, validation history, and final test metrics. Use a real retinal dataset for meaningful Grad-CAM results; the included demo images are only software smoke tests.

Source code uses the `src/retinax/` package layout. Install the project with `pip install -e .` before running the CLI.

## Web interface

Install Streamlit and launch the research UI:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal, upload a retinal image, and optionally provide a trained checkpoint. Without a checkpoint, the UI runs an untrained preview only.

Use `--model baseline`, `--model cnn_lstm`, or `--model moe`. The code is deliberately usable on CPU for smoke tests and Colab GPU runs. It does not ship data, credentials, or checkpoints.

This is a research prototype, not a medical device or diagnostic substitute.

For a complete handoff and setup sequence, see [SETUP.md](SETUP.md).
