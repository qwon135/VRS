
from glob import glob
import pydicom, cv2
import nibabel as nib
from scipy.ndimage import label, center_of_mass
import numpy as np
from skimage.morphology import skeletonize, dilation
import numpy as np
from skimage.measure import marching_cubes
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
import os
from tqdm import tqdm
from nibabel.nicom.dicomwrappers import wrapper_from_file, wrapper_from_data
import pandas as pd

def get_3d_bounding_box(mask):        
    indices = np.argwhere(mask == 1)
    
    xmin, ymin, zmin = indices.min(axis=0)
    xmax, ymax, zmax = indices.max(axis=0)
    
    return xmin, xmax, ymin, ymax, zmin, zmax

def rotate_img(image):
    height, width = image.shape[:2]

    # 회전 중심점 계산
    center = (width // 2, height // 2)

    # 45도 시계방향으로 회전 변환 행렬 생성
    angle = 90
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # 회전 적용
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

from skimage.measure import label
import numpy as np

def get_large_tumor(tumor_mask):
    # tumor_mask는 3D 마스크 이미지라고 가정합니다.

    distal_tumor_mask = dilation(tumor_mask, np.ones((3, 3, 3)))
    labeled_mask = label(distal_tumor_mask)

    # 각 레이블의 크기 계산 (0번 레이블 제외)
    label_counts = np.bincount(labeled_mask.flatten())[1:]

    # 가장 큰 레이블 선택
    largest_label = np.argmax(label_counts) + 1

    # 가장 큰 레이블에 해당하는 영역만 마스크로 만들기
    largest_mask = (labeled_mask == largest_label)
    return largest_mask

def get_3d_dicom(series_path):
    images = sorted(glob(f'{series_path}/*.dcm'))
    # images = np.stack( [pydicom.dcmread(i).pixel_array for i in images], -1)
    images = np.stack( [standardize_pixel_array(pydicom.dcmread(i)) for i in images], -1)
    
    return images

def standardize_pixel_array(dcm) -> np.ndarray:
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
    try:
        center = int(dcm.WindowCenter[0])
        width = int(dcm.WindowWidth[0])
    except:
        center = int(dcm.WindowCenter)
        width = int(dcm.WindowWidth)
    
    low = -1000
    high = 1000
    
    pixel_array = (pixel_array * slope) + intercept
    pixel_array = np.clip(pixel_array, low, high)

    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-6)    
    pixel_array = (pixel_array * 255).astype(np.uint8)
    return pixel_array    

def get_3d_dicom(series_path):
    images = sorted(glob(f'{series_path}/*.dcm'))
    # images = np.stack( [pydicom.dcmread(i).pixel_array for i in images], -1)
    images = np.stack( [standardize_pixel_array(pydicom.dcmread(i)) for i in images], -1)
    
    return images

def vessel_point(vessel_mask, min_size=3, max_gap=2):
    data = [(idx, vessel_mask[:,:,idx].sum()) for idx in range(vessel_mask.shape[-1])]
    regions = []
    start = None
    for i, (idx, value) in enumerate(data):
        if value > 0:
            if start is None:
                start = i
        elif start is not None:
            if i - start >= min_size:
                regions.append((start, i-1))
            start = None

    if start is not None and len(data) - start >= min_size:
        regions.append((start, len(data)-1))

    # Merge regions with small gaps
    merged_regions = []
    for region in regions:
        if not merged_regions or region[0] - merged_regions[-1][1] > max_gap:
            merged_regions.append(region)
        else:
            merged_regions[-1] = (merged_regions[-1][0], region[1])

    # Find the longest region
    if merged_regions:
        longest_region = max(merged_regions, key=lambda x: x[1] - x[0])
        return longest_region
    return None

def get_crop_image_from_tumor_point(images, tumor_mask, vessel_mask, bbox):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    # tumor_point = (int((xmin+xmax)/2), int((ymin+ymax)/2), int((zmin+zmax)/2))    
    # crop_images = images[xmin:xmax, ymin:ymax, zmin:zmax]

    target_wh = 224
    target_z = 48
    xmin = max(xmin-(target_wh//2), 0)
    xmax = xmin + target_wh

    ymin = max(ymin-(target_wh//2), 0)
    ymax= ymin + target_wh

    zmin,zmax = vessel_point(vessel_mask)

    return images[xmin:xmax, ymin:ymax, zmin:zmax], tumor_mask[xmin:xmax, ymin:ymax, zmin:zmax], vessel_mask[xmin:xmax, ymin:ymax, zmin:zmax]

save_root = f'NSCLC_tumor_vessel_crop'

for pid in tqdm(os.listdir('data/NSCLC_CT')):    
    try:
        os.makedirs(f'{save_root}/{pid}/', exist_ok=True)
        dcm = wrapper_from_file(
                    glob(f'data/NSCLC_CT/{pid}/*.dcm')[0]
                    )
        vessel_mask_nib = nib.load(f'data/NSCLC_CT_vessels/{pid}/lung_vessels.nii.gz')    
        org_ax = nib.aff2axcodes(dcm.affine)
        target_ax = nib.aff2axcodes(vessel_mask_nib.affine)

        orig_ornt = nib.orientations.axcodes2ornt(org_ax)
        target_ornt = nib.orientations.axcodes2ornt(target_ax)
        transform = nib.orientations.ornt_transform(orig_ornt, target_ornt)    
        # 변환 행렬 적용
        reoriented_img = vessel_mask_nib.as_reoriented(transform)    
        # 변환된 축 방향 확인
        new_axcodes = nib.aff2axcodes(reoriented_img.affine)
        vessel_mask = reoriented_img.get_fdata()        
        
        images = get_3d_dicom(f'data/NSCLC_CT/{pid}/')    
        tumor_mask = sorted(glob(f'data/NSCLC_tumors/{pid}/*.png'))
        tumor_mask = np.stack([
                        cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in tumor_mask], -1)    
        if not tumor_mask.sum():        
            continue

        tumor_mask = tumor_mask.astype(bool).astype(int)
        tumor_mask = get_large_tumor(tumor_mask)                    
        xmin, xmax, ymin, ymax, zmin, zmax = get_3d_bounding_box(tumor_mask)
        crop_images, tumor_crop, vessel_crop = get_crop_image_from_tumor_point(images, tumor_mask, vessel_mask, (xmin, xmax, ymin, ymax, zmin, zmax))        
        tumor_crop = (tumor_crop.astype(bool).astype(int) * 255).astype('uint8')
        vessel_crop = (vessel_crop.astype(bool).astype(int) * 255).astype('uint8')    

        np.save(f'{save_root}/{pid}/images.npy', crop_images)
        np.save(f'{save_root}/{pid}/tumor_mask.npy', tumor_crop)
        np.save(f'{save_root}/{pid}/vessel_mask.npy', vessel_crop)    
    except:
        print(pid)