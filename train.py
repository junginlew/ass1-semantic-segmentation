import os
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from dataset import CarvanaDataset
from sklearn.model_selection import train_test_split

IMAGE_DIR = "path/to/images"
MASK_DIR  = "path/to/masks"

all_images = sorted(os.listdir(IMAGE_DIR)) #이미지 파일명 리스트
all_masks = sorted(os.listdir(MASK_DIR)) #마스크 파일 명 리스트

train_images, val_images, train__masks, val_masks = train_test_split(
    all_images, all_masks,
    test_size=0.2,
    random_state=42
)

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

val_transform = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

train_dataset=CarvanaDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_list=train_images,
    mask_list=train__masks,
    transform=train_transform
)

val_dataset=CarvanaDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    image_list=val_images,
    mask_list=val_masks,
    transform=val_transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True # 데이터를 GPU로 빠르게 전송하기 위한 임시 보관소 사용
)

val_loader= DataLoader(
    dataset=val_dataset, 
    batch_size=16, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)