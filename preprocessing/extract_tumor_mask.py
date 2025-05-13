import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from tqdm import tqdm
import torch, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from matplotlib import pyplot as plt
from timm.utils import ModelEma
from segmentation_models_pytorch.metrics import iou_score
import segmentation_models_pytorch_custom as smpc
import argparse

class CTumorDataset(Dataset):
    def __init__(self, image_list, transforms=None):
        self.image_list  = image_list
        self.transforms = transforms        
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_path= self.image_list[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)        
        image = self.transforms(image=image)['image']        
        image = image / 255.0
        return image, image_path
    
transform = A.Compose([
        A.Resize(512, 512, cv2.INTER_LINEAR),
        ToTensorV2()
    ], p=1.0)    

class CONFIG:
    epochs = 30
    lr = 5e-5
    ema_decay = 0.995
    weight_decay=5e-2
    epochs = 30
    fold = 0
    backbone = 'convnext_tiny' 
    device = 'cuda:0'
    IMG_SIZE = 512

def build_model(args=None):
    model = smpc.Unet(
        encoder_name=args.backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_args={"in_channels": 1},
        encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,        # model output channels (number of classes in your dataset)
        activation=None,
        # decoder_norm_type="GN",
        # decoder_act_type="GeLU",
    )
    model.to(args.device)
    return model

def main(args):
    model = build_model(args)
    model.load_state_dict(torch.load(f'ckpt/convnext_tiny_imgsize512.pt', map_location='cpu'), 'cpu')
    model = model.eval()

    image_list = glob(f'NSCLC_CT_png/*/*/*.png')
    infer_datatset = CTumorDataset(image_list, transform)
    infer_loader = DataLoader(infer_datatset, batch_size=16, num_workers=16, shuffle=False)

    threshold = 0.3
    for images, image_paths in tqdm(infer_loader):
        with torch.no_grad():
            logits = []            
            logits.append(model(images.to(args.device)))

        logits = torch.stack(logits, 0)
        logits = logits.mean(0).sigmoid().cpu().squeeze(1).numpy()

        for mask, image_path in zip(logits, image_paths):
            mask = ((mask > threshold).astype(int) * 255).astype('uint8')
            pid = image_path.split('/')[-3]
            series = image_path.split('/')[-2]
            fname = image_path.split('/')[-1]

            path =  f'NSCLC_CT_png/{pid}/{series}'
            os.makedirs(path, exist_ok=True)

            cv2.imwrite(f'{path}/{fname}', mask)
            

# 사용 예시:
if __name__ == "__main__":
    args = CONFIG()
    main(args)