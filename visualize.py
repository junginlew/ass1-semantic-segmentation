import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNet

TEST_DIR = "./data/test"
MODEL_PATH = "best_unet_model.pth"
NUM_IMAGES = 5

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True)) #저장된 모델 가중치 불러오기, weights_only=True로 보안 강화
    model.eval()
    print(f"Loaded pretrained model from '{MODEL_PATH}'")
    
    test_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    
    test_images = sorted(os.listdir(TEST_DIR))[:NUM_IMAGES] #test 이미지 파일명 리스트 상위 NUM_IMAGES개
    
    plt.figure(figsize=(10,5*NUM_IMAGES))
    
    with torch.no_grad():
        for i, img_name in enumerate(test_images):
            img_path = os.path.join(TEST_DIR, img_name)
            image_bgr = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_viz = A.Resize(height=256, width=256)(image=image_rgb)["image"] #시각화용 원본 이미지 크기 통일
            
            # 모델에 넣을 입력 텐서
            transformed = test_transform(image=image_rgb)
            input_tensor = transformed["image"]
            input_tensor = input_tensor.unsqueeze(0).to(device) #Batch 차원 추가, GPU로 이동
            
            output = model(input_tensor) #모델 추론
            
            pred = torch.sigmoid(output) #Logit을 0~1 사이의 확률로 변환
            pred_mask = (pred > 0.5).float() #이진 마스크 텐서
            
            pred_mask_np = pred_mask.squeeze().cpu().numpy() # 2차원으로 변환, GPU에서 CPU로 이동, Numpy 배열로 변환

            # Matplotlib 시각화
            # 좌측: 원본 이미지
            plt.subplot(NUM_IMAGES, 2, i*2 + 1)
            plt.imshow(image_viz)
            plt.title(f"Test Image: {img_name}")
            plt.axis('off')

            # 우측: 모델이 예측한 흑백 마스크
            plt.subplot(NUM_IMAGES, 2, i*2 + 2)
            plt.imshow(pred_mask_np, cmap='gray')
            plt.title("Predicted Mask (U-Net)")
            plt.axis('off')
            
    plt.tight_layout() #레이아웃 조정
    plt.savefig("test_visualization.png")
    print("예측 시각화 결과가 'test_visualization.png'에 저장되었습니다.")      
            