# Deep‑Learning Vascular Risk Score (VRS) for ICI Response in NSCLC
---

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

1. **Raw CT scans** (DICOM/NIfTI) placed in `data/` 
2. Run lung vessel, tumor mask extraction:

```bash
python preprocessing/preprocessing_dicom.py 
python preprocessing/extract_vessel_mask.py 
python preprocessing/extract_tumor_mask.py 
python preprocessing/extract_tumor_vessel_image.py 
```

## 4. inference VRS score

View `VRS.ipynb``