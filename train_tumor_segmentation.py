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
    def __init__(self, df, transforms=None, mode='train'):
        self.df  = df
        self.transforms = transforms
        self.mode = mode
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        image_path= self.df.loc[idx, 'image_path']
        mask_path = self.df.loc[idx, 'mask_path']

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        aug_data = self.transforms(image=image, mask=mask)
        
        image  = aug_data['image'] / 255.0       
        mask  = aug_data['mask'] 
        mask[mask<128] = 0
        mask[mask>=128] = 255
        mask = mask / 255.0
                
        return image, mask
    
class METRICS:
    def __init__(self):
        self.tp_list, self.fp_list, self.fn_list, self.tn_list = [], [], [], []

    def add_prediction(self, preds, masks):
        tp, fp, fn, tn = smp.metrics.get_stats(preds.cpu(), masks.cpu().unsqueeze(1).long(), mode='binary', threshold=0.5)
        self.tp_list.append(tp)
        self.fp_list.append(fp)
        self.fn_list.append(fn)
        self.tn_list.append(tn)
    
    def get_score(self):
        if type(self.tp_list) == list:    
            self.tp_list, self.fp_list, self.fn_list, self.tn_list = torch.cat(self.tp_list, 0), torch.cat(self.fp_list, 0), torch.cat(self.fn_list, 0),torch.cat(self.tn_list,0)            
        iou_score = smp.metrics.iou_score(self.tp_list, self.fp_list, self.fn_list, self.tn_list, reduction='macro')
        f1_score = smp.metrics.f1_score(self.tp_list, self.fp_list, self.fn_list, self.tn_list, reduction='macro')
        precision = smp.metrics.precision(self.tp_list, self.fp_list, self.fn_list, self.tn_list, reduction='macro')
        recall = smp.metrics.recall(self.tp_list, self.fp_list, self.fn_list, self.tn_list, reduction='macro')
        return iou_score.item(), f1_score.item(), precision.item(), recall.item()
    
def get_transform(args):
    train_transform = A.Compose([
            A.Resize(args.IMG_SIZE, args.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),        
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.3),
                A.ElasticTransform(p=0.1)
                ],
                p=0.3),
            A.OneOf([
                A.RandomGamma(gamma_limit=(40, 80), p=0.15),
                A.Blur(blur_limit=(3,7), p=1.0),
                A.GaussNoise(p=1.0, var_limit=(30.0, 50.0)),
                A.MultiplicativeNoise(p=1.0, multiplier=(0.9, 1.1), elementwise=True)
            ], p=0.1),
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.3, rotate_limit=20, p=0.3),        
            A.CoarseDropout(max_holes=3, min_holes=1, p=0.5, max_height=int(args.IMG_SIZE * 0.05), max_width=int(args.IMG_SIZE * 0.05), mask_fill_value=None),
            ToTensorV2()
                ], p=1.0)


    valid_transform = A.Compose([
            A.Resize(args.IMG_SIZE, args.IMG_SIZE, cv2.INTER_LINEAR),
            ToTensorV2()
        ], p=1.0)
    return train_transform, valid_transform

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def get_data(fold):    
    
    data = pd.DataFrame( {'image_path' : sorted(glob(f'/data1/02_pjh/CT_lidc/ct_png_1000to1000/*/*.png'))})
    data['mask_path'] = data['image_path'].str.replace('ct_png_1000to1000', 'binary_mask')

    data['patient_id'] = data['image_path'].str.split('/').str[5]

    kfold = KFold(n_splits=5, )    
    patientid_list = data['patient_id'].unique()

    for k, (t_idx, v_idx) in enumerate(kfold.split(patientid_list)):
        train_pid, valid_pid = patientid_list[t_idx], patientid_list[v_idx]

        train_df = data[data['patient_id'].isin(train_pid)].reset_index(drop=True)
        valid_df = data[data['patient_id'].isin(valid_pid)].reset_index(drop=True)
        if k == fold:
            break
    return train_df, valid_df

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

def down_sampling(train_df):
    mask_gt = pd.read_csv(f'mask_gt.csv', index_col=None)

    train_non_gt = train_df[train_df['mask_path'].isin(mask_gt[mask_gt['is_gt'] != 1]['mask_path'].values)].reset_index(drop=True)
    train_gt = train_df[train_df['mask_path'].isin(mask_gt[mask_gt['is_gt'] == 1]['mask_path'].values)].reset_index(drop=True)

    train_non_gt = train_non_gt.sample(train_gt.shape[0] * 1).reset_index(drop=True)
    down_train_df = pd.concat([train_gt, train_non_gt]).reset_index(drop=True)
    return down_train_df

def main(args):

    train_df, valid_df = get_data(args.fold)
    train_transform, valid_transform = get_transform(args)
    model = build_model(args=args)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    dce_loss_fn = DiceLoss(mode='binary')
    def loss_fn(x, y):
        return dce_loss_fn(x,y) * 0.5 + bce_loss_fn(x,y) * 0.5

    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=False)
    ema = ModelEma(model, decay=args.ema_decay)

    best_loss = 1e6
    for epoch in range(args.epochs):    
        train_datatset = CTumorDataset(down_sampling( train_df), train_transform)
        valid_datatset = CTumorDataset(valid_df, valid_transform)

        train_loader = DataLoader(train_datatset, batch_size=args.batch_size, num_workers=8, shuffle=True)
        valid_loader = DataLoader(valid_datatset, batch_size=args.batch_size, num_workers=8, shuffle=False)

        train_loss = 0
        model.train()
        metric = METRICS()
        for images, masks in train_loader:
            preds = model(images.to(args.device))
            optimizer.zero_grad()
            loss = loss_fn(preds.squeeze(1), masks.to(args.device))
            loss.backward()
            optimizer.step()    
            ema.update(model)

            train_loss += loss.cpu().item()
        train_loss /= len(train_loader)
        lr_scheduler.step()

        valid_loss = 0
        model.eval()
        
        for images, masks in valid_loader:
            with torch.no_grad():
                preds = model(images.to(args.device))
                loss = loss_fn(preds.squeeze(1), masks.to(args.device))
            valid_loss += loss.cpu().item()
            metric.add_prediction(preds, masks)
        iou_score, f1_score, precision, recall = metric.get_score()

        valid_loss /= len(valid_loader)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'tumor_model_ckpt/{args.backbone}_imgsize{args.IMG_SIZE}_{args.fold}.pt')
        print(f'EPOCH : {epoch} | train_loss : {train_loss:.4f} | valid_loss : {valid_loss:.4f} | iou : {iou_score:.4f} | f1 : {f1_score:.4f} | prc : {precision:.4f} | rec : {recall:.4f}') 
    print()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--backbone", type=str, default='convnext_tiny')
    parser.add_argument("--IMG_SIZE", type=int, default=512)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)    
    main(args)    