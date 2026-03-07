import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from dataset import CarvanaDataset

train_transform = A.Compose(
    [
        A.Resize(height=256, width=256),  # 이미지 크기 통일
        A.HorizontalFlip(p=0.5),  # 데이터 증강, 50%확률로 좌우 반전
        A.Normalize(
            #ImageNet 평균, 표준편차
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,  # /255.0 로 0~1로 맞춤
        ),  # Normalize는 image만 적용. mask는 적용 X
        ToTensorV2(), #Pytorch Tensor로 변환, 차원 순서 변경(C,H,W)
    ]
)

train_dataset=CarvanaDataset(
    image_dir="image folder path",
    mask_dir="mask folder path",
    transform=train_transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True # 데이터를 GPU로 빠르게 전송하기 위한 임시 보관소 사용
)

'''
dataset = NumpySegDataset(images_path, masks_path)
total_len = len(dataset)
train_len = int(total_len * 0.8)
val_len = total_len - train_len

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
'''