import pandas as pd
from glob import glob
from tqdm import tqdm
import os, cv2
from pydicom import dcmread
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np

def standardize_pixel_array(dcm, low, high) -> np.ndarray:
    """
    Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
    """
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype 
        pixel_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
#         pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(new_array, dcm)

    intercept = float(dcm.RescaleIntercept)
    slope = float(dcm.RescaleSlope)    
    
    pixel_array = (pixel_array * slope) + intercept
    pixel_array = np.clip(pixel_array, low, high)

    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-6)

    if dcm.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = 1 - pixel_array
    pixel_array = (pixel_array * 255).astype(np.uint8)
    return pixel_array

ct_path = 'data/NSCLC_CT'
save_root = 'data/NSCLC_CT_png'
# os.mkdir(save_root)

for pid in tqdm(os.listdir(ct_path)):
    pid_path = f'{ct_path}/{pid}'    
    os.makedirs(f'{save_root}/{pid}', exist_ok=True)
    
    for series in os.listdir(f'{pid_path}'):
        series_path = f'{pid_path}/{series}'
        os.makedirs(f'{save_root}/{pid}/{series}', exist_ok=True)

        for dcm_name in os.listdir(series_path):
            dcm_path = f'{series_path}/{dcm_name}'
            dcm_object = dcmread(dcm_path)
            
            image = standardize_pixel_array(dcm_object, low=-1000, high=1000)
            cv2.imwrite(f'{save_root}/{pid}/{series}/{dcm_name.replace(".dcm", "")}.png', image)