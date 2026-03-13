import os
import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import UNet
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from dataset import CarvanaDataset
from sklearn.model_selection import train_test_split

IMAGE_DIR = "./data/train"
MASK_DIR  = "./data/train_masks"

def calculate_metrics(outputs,masks,smooth=1e-6):
    preds = torch.sigmoid(outputs) #0~1로 변환
    preds = (preds > 0.5).float() #0, 1 로 이진화
    
    #1차원으로 변환
    preds=preds.view(-1) 
    masks=masks.view(-1)
    
    intersection = (preds * masks).sum()
    total = preds.sum() + masks.sum()
    union = total - intersection
    
    dice = (2. * intersection + smooth) / (total + smooth) #smooth는 분모 0 방지
    iou = (intersection + smooth) / (union + smooth)
    
    return dice.item(), iou.item()
    
    
if __name__ == '__main__':

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
    
    #학습률 스케줄러 (검증 Loss가 3번의 에폭 동안 안 떨어지면 lr을 절반으로 줄임)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',      # Loss를 최소화하는 방향
        factor=0.5,      # 학습률을 감소시킬 비율 (기존 lr * 0.5)
        patience=3,      # Loss가 개선되지 않아도 참는 에폭 수
        min_lr=1e-6      # 학습률이 떨어질 수 있는 하한선
    )
    
    history_train_loss = []
    history_val_loss = []
    history_lr = []

    num_epochs = 30
    best_val_iou = 0.0

    # Training loop
    for epoch in range(num_epochs):
        
        #Train 모드
        model.train()
        train_loss, train_dice, train_iou = 0.0, 0.0, 0.0
        
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
            with torch.no_grad():
                dice, iou = calculate_metrics(outputs, masks)
                train_dice += dice
                train_iou += iou
            
        #Validation 모드
        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        
        with torch.no_grad(): #검증 단계에서는 기울기 계산 X
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                dice, iou = calculate_metrics(outputs, masks)
                val_dice += dice
                val_iou += iou
                
        #에폭마다 평균 손실 출력
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        # 스케줄러 업데이트
        scheduler.step(avg_val_loss) # 검증 로스를 기준으로 학습률을 조절
        current_lr = optimizer.param_groups[0]['lr'] # 현재 업데이트된 학습률
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)
        history_lr.append(current_lr)
  
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  [Train] Loss: {avg_train_loss:.4f} | Dice: {avg_train_dice:.4f} | IoU: {avg_train_iou:.4f}")
        print(f"  [Valid] Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f} | IoU: {avg_val_iou:.4f}")
        print(f"  [ LR  ] Current Learning Rate: {current_lr:.6f}")
        
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            save_file(model.state_dict(), "best_unet_model.safetensors") #모델의 가중치 저장, safetensors는 보안 강화된 모델 저장 형식
            print(f"  Best model saved! (IoU: {best_val_iou:.4f})")
        print("-" * 50)
        
    #학습 곡선 시각화
    plt.figure(figsize=(12, 5))

    # Train, Val Loss
    plt.subplot(1, 2, 1)
    plt.plot(history_train_loss, label='Train Loss')
    plt.plot(history_val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Learning Rate Decay
    plt.subplot(1, 2, 2)
    plt.plot(history_lr, label='Learning Rate', color='green', marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Convergence')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("loss_and_lr_curve.png")
    print("Graph saved to 'loss_and_lr_curve.png'.")
        