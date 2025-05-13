from totalsegmentator.python_api import totalsegmentator
from glob import glob
import pydicom, cv2
from tqdm import tqdm
import os
import nibabel as nib
from matplotlib import pyplot as plt
import pandas as pd
import argparse

def main():
    save_root = 'NSCLC_vessels'
    df = pd.read_csv('vessel_ct_series.csv', index_col=None)    
    
    for series_path in tqdm(df['series_path']):
        series = series_path.split('/')[-1]
        series_newname = series.split('|')[0]
        pid = series_path.split('/')[-2]
                
        os.makedirs(f'{save_root}/{pid}/{series_newname}', exist_ok=True)
        try:
            seg_img = totalsegmentator(f'{series_path}', f'{save_root}/{pid}/{series_newname}', task='lung_vessels',fast=False, skip_saving=False, device='cuda')
        
        except:
            print(series_path)
            continue

if __name__ == '__main__':    
    main()    