# Deep‑Learning Vascular Risk Score (VRS) for ICI Response in NSCLC
---
This repository provides code to compute a Deep Learning-based Vascular Remodeling Score (VRS) from CT images, designed to predict ICI response in NSCLC patients.

## 1. Repository layout

```
├── models/
├── preprocessing/
│   ├── extract_tumor_mask/
│   ├── extract_tumor_vessel_image.py/
│   ├── extract_vessel_mask.py/
│   └── preprocessing_dicom.py/
├── train_tumor_segmentation.py/
├── train_tumor_vessel_features.py/
└── TumorVesselDataset.py/
 
```

## 2. Setup

```bash
# create environment (Python ≥3.10)
conda create -n vrs python=3.10 -y
conda activate vrs
pip install torch torchvision
pip install pydicom

```

> **Key dependencies**: PyTorch ≥1.13, TorchVision ≥0.14, pydicom

## 3. Data preparation

We train a tumor segmentation model by running `train_tumor_segmentation.py`.

For vessel segmentation, we use a pre-trained total segmentation model.

1. **Raw CT scans** (DICOM/NIfTI) placed in `data/` 
2. Run lung vessel, tumor mask extraction:

```bash
python preprocessing/preprocessing_dicom.py 
python preprocessing/extract_vessel_mask.py 
python preprocessing/extract_tumor_mask.py 
python preprocessing/extract_tumor_vessel_image.py 
```

## 4. Train Vessel Features

We train vessel-related features using contrastive learning to capture the heterogeneity between normal and tumor vasculature.

```bash
python train_tumor_vessel_features.py 
```
## 5. inference VRS score
Run all cells in `VRS.ipynb`. It loads extracted features, applies the GMM, and visualizes the resulting VRS scores.

View `VRS.ipynb`

The VRS (Vascular Remodeling Score) is inferred using a Gaussian Mixture Model (GMM), which quantifies the complexity of tumor vasculature based on deviations from the distribution of normal vascular features.