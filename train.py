import os
import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
from model import UNet
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from dataset import CarvanaDataset
from sklearn.model_selection import train_test_split

IMAGE_DIR = "path/to/images"
MASK_DIR  = "path/to/masks"

all_images = sorted(os.listdir(IMAGE_DIR)) #이미지 파일명 리스트
all_masks = sorted(os.listdir(MASK_DIR)) #마스크 파일 명 리스트

train_images, val_images, train_masks, val_masks = train_test_split(
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
    mask_list=train_masks,
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #GPU 사용 설정
model = UNet(in_channels=3, out_channels=1).to(device) #UNet 모델 생성, GPU로 이동
criterion = nn.BCEWithLogitsLoss() #Sigmoid(0~1로 변환) + BCE 연산
optimizer = optim.Adam(model.parameters(), lr=1e-4) #모델의 가중치 업데이트, 학습률 0.0001

num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    
    #Train 모드
    model.train()
    train_loss = 0.0
    
    for images, masks in train_loader:
        #GPU로 데이터 이동
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad() #기울기 초기화
        outputs = model(images) #순전파, 마스크 예측
        loss = criterion(outputs, masks) #손실 계산
        loss.backward() #역전파, 가중치 수정을 위한 기울기 계산
        optimizer.step() #가중치 업데이트
        
        train_loss += loss.item()
        
    #Validation 모드
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad(): #검증 단계에서는 기울기 계산 X
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
    #에폭마다 평균 손실 출력
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    