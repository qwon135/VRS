import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import MinMaxScaler, Binarizer
from mamba_ssm import Mamba
import numpy as np
import pickle, argparse
from matplotlib import pyplot as plt
from TumorVesselDataset import CustomDataset
import os
from tqdm import tqdm
import cv2
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score,confusion_matrix, f1_score, roc_auc_score
from segmentation_models_pytorch.losses import FocalLoss
from sklearn.model_selection import train_test_split, StratifiedKFold
from pydicom import dcmread
import shutil
import argparse
import transformers
from torch_ema import ExponentialMovingAverage
from glob import glob
from pycox.models.loss import cox_ph_loss, cox_cc_loss, nll_logistic_hazard, cox_ph_loss_sorted
import random, timm
from models.feature_extractor import CT25D

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(2023)    

def get_transform(IMG_SIZE):
    train_transform = A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.CoarseDropout(always_apply=False, p=0.5, max_holes = 1,max_height = 16, max_width = 16, min_holes = 1,min_height = 8,min_width = 8),
        A.Perspective(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(p=0.5, limit=(-25, 25)),    
        ToTensorV2(), 
            ], additional_targets={
                'mask1': 'mask',
                'mask2': 'mask'
            })
    valid_transform = A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        ToTensorV2()
            ], additional_targets={
                'mask1': 'mask',
                'mask2': 'mask'
            })
    return train_transform, valid_transform

def trainer(model, optimizer, criterion,  criterion_cox, ema, train_loader, device, epoch):
    train_loss = 0.0
    model.train()

    for img, label in train_loader:
        OS, dead = label                
        OS, dead = OS.to(device), dead.to(device)        

        img = img.to(device)
        optimizer.zero_grad()
        pred_os = model(img).squeeze(-1)        
        
        dead_cox_loss = criterion_cox(pred_os, OS, dead) /img.shape[0]
        loss = dead_cox_loss
        train_loss += loss.item()

        loss.backward()
        grad_=torch.nn.utils.clip_grad_norm_(model.parameters(), 50)

        optimizer.step()
        ema.update()    

    train_loss /=len(train_loader)    
    return train_loss

def validator(model, scheduler, criterion,  criterion_cox, valid_loader, device):
    valid_loss = 0.0    
    model.eval()
    valid_risk_score = []

    for img, label in valid_loader:
        OS, dead = label                
        OS, dead = OS.to(device), dead.to(device)
        img = img.to(device)

        with torch.no_grad():
            pred_os = model(img).squeeze(-1)                        
            dead_cox_loss = criterion_cox(pred_os, OS, dead) /img.shape[0]
                    
        loss = dead_cox_loss        
        valid_risk_score += pred_os.cpu().tolist()        
        
        valid_loss += loss.item()
        

    valid_loss /= len(valid_loader)        

    scheduler.step()
    return valid_risk_score, valid_loss

def get_loader(df, tform):
    class_w = 1 / np.bincount(df['dead'])
    cw = [class_w[int(i)] for i in df['dead'].tolist()]
    dset = CustomDataset(df, transform=tform, mode='test')
    sampler = WeightedRandomSampler(weights=cw, num_samples=len(dset))#(dset, df['dead'].values, args.batch_size)
    loader = DataLoader(dset,
                                sampler=sampler,
                                batch_size=args.batch_size,  drop_last=False , shuffle=False, num_workers = args.num_workers)
    return loader

def main(args):
    model = CT25D().to(args.device)

    df = pd.read_csv('hypofractionated_radiotherapy.csv', index_col=None)
    loss_fn = nn.BCEWithLogitsLoss()

    param_groups = [
        {'params': [], 'lr':  args.cnn_lr, 'weight_decay' : args.weight_decay},
        {'params': [], 'lr': args.seq_lr, 'weight_decay' : args.weight_decay},
        {'params': [], 'lr': args.seq_lr, 'weight_decay' : args.weight_decay}
    ]    
    for name, param in model.named_parameters():
        if  'cnn' in name:
            param_groups[0]["params"].append(param) # convs가 이름에 포함되면 첫번째 그룹에 추가
        elif 'lstm' in name:
            param_groups[1]["params"].append(param) # convs가 이름에 포함되면 첫번째 그룹에 추가
        else:
            param_groups[2]["params"].append(param) # 아니면 두번째 그룹에 추가

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max =10, eta_min= 1e-7)
    best_loss = 1e6

    train_df, valid_df = train_test_split(df, stratify=df['relapse'].astype(str), test_size=0.2, random_state=args.seed)
    train_df, valid_df = train_df.reset_index(drop=True), valid_df.reset_index(drop=True)
    
    loss_fn_cox = cox_ph_loss_sorted
    train_transform, valid_transform = get_transform([args.IMG_SIZE, args.IMG_SIZE])

    train_loader = get_loader(train_df, train_transform)
    valid_loader = get_loader(valid_df, valid_transform)

    for epoch in range(args.epochs):
        train_cox_loss = trainer(model, optimizer, loss_fn, loss_fn_cox, ema, train_loader, args.device, epoch)     
        valid_risk_score, valid_loss = validator(model, scheduler, loss_fn, loss_fn_cox, valid_loader, args.device)    

        print(f'EPOCH : {epoch} | t_loss : {train_cox_loss:.4f} | v_loss : {valid_loss:.4f}')

        if valid_loss < best_loss:
            print(f'EPOCH : {epoch} is best loss!!')
            torch.save(model.state_dict(), f'ckpt/best_RT.pt')
            best_loss = valid_loss
            print()


def get_args():
    parser = argparse.ArgumentParser(description='Tumor Vessel Analysis Configuration')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=777,
                        help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--fold', type=int, default=1,
                        help='Fold number for cross validation')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--warmup_epoch', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to use for training (e.g., cuda:0, cuda:1, cpu)')
    
    # Loss and optimization
    parser.add_argument('--loss_type', type=str, default='BCE',
                        help='Type of loss function')
    parser.add_argument('--cnn_lr', type=float, default=1e-6,
                        help='Learning rate for CNN')
    parser.add_argument('--seq_lr', type=float, default=1e-5,
                        help='Learning rate for sequence model')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay for optimization')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='Decay rate for EMA')
    parser.add_argument('--step', type=int, default=18,
                        help='Step size for learning rate schedule')
    parser.add_argument('--interval', type=int, default=-1,
                        help='Interval parameter')
    parser.add_argument('--optimizer_type', type=str, default='adamw',
                        help='Type of optimizer to use')
    
    # Data
    parser.add_argument('--root', type=str, 
                        default='data/NSCLC_tumor_vessel',
                        help='Root directory for data')

    args = parser.parse_args()
    return args

# 사용 예시:
if __name__ == "__main__":
    args = get_args()
    main(args)