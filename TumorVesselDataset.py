from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

def get_center_indices(length, step, interval=1):
    center = length // 2
    start = center - (step // 2) * interval
    end = center + (step // 2) * interval + (1 if step % 2 != 0 else 0)
    indices = list(range(start, end, interval))
    
    # 인덱스가 유효 범위를 벗어나지 않도록 조정
    indices = [max(0, min(i, length - 1)) for i in indices]
    return indices[:step]  # step 개수만큼만 반환

class CustomDataset(Dataset):
    def __init__(self, df, transform, mode, args):
        self.df = df
        self.transform = transform
        self.mode = mode
        self.args = args

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):        
        oid = self.df.loc[idx, 'patientid']        
        
        ct_images = np.load( f'{self.args.root}/{oid}/images.npy')
        tumor_mask = np.load( f'{self.args.root}/{oid}/tumor_mask.npy')
        vessel_mask = np.load( f'{self.args.root}/{oid}/vessel_mask.npy')

        tumor_value = np.stack([tumor_mask[:, :, i].sum() for i in range(tumor_mask.shape[-1])])
        tumor_idx = tumor_value.argmax()
        if tumor_idx:            
            end = min(tumor_mask.shape[-1], tumor_idx * 2)
            start = tumor_idx * 2 - end
            ct_images = ct_images[:,:,start:end]
            tumor_mask = tumor_mask[:,:,start:end]
            vessel_mask = vessel_mask[:,:,start:end]
        images = []
        v_masks = []
        for img_idx in range(ct_images.shape[-1]):
            img = ct_images[:,:,img_idx]
            t_msk = tumor_mask[:,:,img_idx]
            v_msk = vessel_mask[:,:,img_idx]
                        
            msk = v_msk.astype(bool).astype(int)
            t_msk = (~(t_msk.astype(bool))).astype(int)
            img = (img * t_msk).astype('uint8')
            img = (img * msk).astype('uint8')

            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            transformed = self.transform(image=img)
            images.append(transformed['image'] / 255)
            v_masks.append(self.transform(image=v_msk)['image']/255)
        
        step_idx = get_center_indices(ct_images.shape[-1], self.args.step, interval=self.args.interval)

        images = torch.stack([images[i] for i in step_idx], 0)  # [Seq, Channel, W, H]
        v_masks = torch.stack([v_masks[i] for i in step_idx], 0)  # [Seq, Channel, W, H]

        images = images.reshape(self.args.step//3, 3, images.shape[-1] ,images.shape[-2])
        v_masks = v_masks.reshape(self.args.step//3, 3, images.shape[-1] ,images.shape[-2])

        return images, v_masks
        
class NormalVesselDataset(Dataset):
    def __init__(self, pids, transform, mode, args):
        self.pids = pids
        self.transform = transform
        self.mode = mode
        self.args = args

    def __len__(self):
        return len(self.pids)
    
    def __getitem__(self, idx):
        pid = self.pids[idx]
        ct_images = np.load( f'{self.args.root}/{pid}/images.npy')
        vessel_mask = np.load( f'{self.args.root}/{pid}/vessel_mask.npy')

        if vessel_mask.sum() == 0:
            zmin, zmax = 0, ct_images.shape[0]-1
        else:
            x,y,z = np.where(vessel_mask != 0)
            xmin, ymin, zmin, xmax, ymax, zmax = min(x), min(y), min(z), max(x), max(y), max(z)

        ct_images = ct_images[:, :, zmin:zmax]
        vessel_mask = vessel_mask[:, :, zmin:zmax]        
        
        images = []
        v_masks = []
        for img_idx in range(ct_images.shape[-1]):
            img = ct_images[:,:,img_idx]
            v_msk = vessel_mask[:,:,img_idx]                    
            msk = v_msk.astype(bool).astype(int)
            img = (img * msk).astype('uint8')
            
            transformed = self.transform(image=img)
            images.append(transformed['image'] / 255)
            v_masks.append(self.transform(image=v_msk)['image']/255)

        
        step_idx = get_center_indices(ct_images.shape[-1], self.args.step, interval=self.args.interval)

        images = torch.stack([images[i] for i in step_idx], 0)  # [Seq, Channel, W, H]
        v_masks = torch.stack([v_masks[i] for i in step_idx], 0)  # [Seq, Channel, W, H]

        images = images.reshape(self.args.step//3, 3, images.shape[-1] ,images.shape[-2])
        v_masks = v_masks.reshape(self.args.step//3, 3, images.shape[-1] ,images.shape[-2])

        return images        