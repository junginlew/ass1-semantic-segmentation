import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class CarvanaDataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, image_list=None, mask_list=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        if image_list is not None and mask_list is not None:
            self.image_list=image_list
            self.mask_list=mask_list
        else:
            self.image_list = sorted(os.listdir(image_dir))
            self.mask_list = sorted(os.listdir(mask_dir))
        self.transform = transform
        assert len(self.image_list) == len(self.mask_list), f"Error: Number of images({len(self.image_list)}) and masks({len(self.mask_list)}) differ."

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        mask_name = self.mask_list[index]        
        
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #Open CV는 BGR, PyTorch는 RGB를 쓰므로 변환
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32) #마스크를 0과 1로 정규화
        
        if self.transform is not None: 
            transformed = self.transform(image=image, mask=mask)  # tensor로 변환
            image = transformed['image']
            mask = transformed['mask']
            
            mask = mask.unsqueeze(0)  #mask 3차원 변환 (1,H,W)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 #텐서로 변환, (C,H,W)로 변환, 0.0 ~ 1.0 사이 값으로 변환
            mask = torch.from_numpy(mask).unsqueeze(0) # 텐서로 변환, (1,H,W)로 변환
            
        return image, mask