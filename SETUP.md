# RetinaX handoff guide

## 1. Install

```bash
cd /path/to/retinax
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Prepare classification data

Use five folders named exactly:

```text
classification_dataset/
├── Mild/
├── Moderate/
├── No_DR/
├── Proliferate_DR/
└── Severe/
```

Verify it:

```bash
python -m retinax.cli inspect-data --data-root "/path/to/classification_dataset"
```

Train and save the classifier:

```bash
python -m retinax.cli train \
  --data-root "/path/to/classification_dataset" \
  --model baseline \
  --epochs 15 \
  --batch-size 32 \
  --checkpoint "checkpoints/retinax_baseline.pt"
```

## 3. Prepare IDRiD segmentation data

Download `A. Segmentation.zip` from the official [IDRiD page](https://idrid.grand-challenge.org/Data/), then use its two folders:

```text
A. Segmentation/
├── 1. Original Images/
└── 2. All Segmentation Groundtruths/
```

Train microaneurysm localization:

```bash
python -m retinax.cli train-segmentation \
  --images "/path/to/A. Segmentation/1. Original Images" \
  --masks "/path/to/A. Segmentation/2. All Segmentation Groundtruths" \
  --lesion MA \
  --epochs 20 \
  --batch-size 4 \
  --checkpoint "checkpoints/retinax_unet_ma.pt"
```

Supported lesion codes are `MA` (microaneurysms), `HE` (hemorrhages), `EX` (hard exudates), and `SE` (soft exudates).

## 4. Run the interface

```bash
streamlit run app.py
```

The sidebar defaults to:

```text
checkpoints/retinax_baseline.pt
checkpoints/retinax_unet_ma.pt
```

Upload a JPG or PNG retinal image and select **Analyse image**. Keep the IDRiD license files with the dataset and cite IDRiD in reports. RetinaX is a research prototype, not a clinical diagnostic device.
