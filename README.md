# Semantic Segmentation 과제 보고서

## 1. 과제 개요

U-Net 아키텍처를 PyTorch로 직접 구현하여 **Carvana Image Masking Challenge** 데이터셋에서 자동차 영역을 픽셀 단위로 분리하는 이진 시맨틱 세그멘테이션(Binary Semantic Segmentation)을 수행한다.

---

## 2. 데이터셋

- **출처**: Kaggle — Carvana Image Masking Challenge
- **입력**: RGB 자동차 이미지 (`.jpg`)
- **레이블**: 이진 마스크 (차량=1, 배경=0)
- **분할**: `sklearn.train_test_split` — Train 80% / Validation 20% (`random_state=42`)

---

## 3. 전처리 및 데이터 증강

`albumentations` 라이브러리로 파이프라인을 구성하였다.

| 단계 | 적용 변환 |
|------|----------|
| Train | `Resize(256×256)` → `HorizontalFlip(p=0.5)` → `Normalize(ImageNet)` → `ToTensorV2` |
| Validation/Test | `Resize(256×256)` → `Normalize(ImageNet)` → `ToTensorV2` |

- **ImageNet 통계**: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- Normalize는 이미지에만 적용되며, 마스크는 `(mask > 0).astype(float32)`로 이진화만 수행
- 검증·추론 시에는 HorizontalFlip 등 데이터 증강 미적용

---

## 4. 모델 아키텍처 — U-Net

### 4.1 기본 블록: DoubleConv

```
Conv2d(in_ch → out_ch, 3×3, padding=1) → BatchNorm2d → ReLU
Conv2d(out_ch → out_ch, 3×3, padding=1) → BatchNorm2d → ReLU
```

`padding=1`로 합성곱 전후 공간 크기(H, W)를 유지한다.

### 4.2 전체 구조

```
Input: (B, 3, 256, 256)

[Encoder]
  down1: DoubleConv(3→64)   + MaxPool → (B, 64,  128, 128)  [skip1]
  down2: DoubleConv(64→128) + MaxPool → (B, 128,  64,  64)  [skip2]
  down3: DoubleConv(128→256)+ MaxPool → (B, 256,  32,  32)  [skip3]
  down4: DoubleConv(256→512)+ MaxPool → (B, 512,  16,  16)  [skip4]

[Bottleneck]
  DoubleConv(512→1024)               → (B, 1024, 16,  16)

[Decoder]
  ConvTranspose2d + cat(skip4) → DoubleConv(1024→512)  → (B, 512, 32, 32)
  ConvTranspose2d + cat(skip3) → DoubleConv(512→256)   → (B, 256, 64, 64)
  ConvTranspose2d + cat(skip2) → DoubleConv(256→128)   → (B, 128,128,128)
  ConvTranspose2d + cat(skip1) → DoubleConv(128→64)    → (B,  64,256,256)

[Output]
  Conv2d(64→1, 1×1)                  → (B, 1, 256, 256)  ← logit
```

### 4.3 Skip Connection

인코더 각 단계의 특성맵(`skip1~4`)을 디코더에서 채널 축(`dim=1`)으로 concatenate한다. 다운샘플링 과정에서 손실된 공간 정보(경계, 질감)를 복원에 직접 전달하는 역할을 한다.

### 4.4 출력 설계

최종 레이어는 1×1 합성곱으로 채널을 1로 줄인 **logit**을 출력한다. 학습 시에는 `BCEWithLogitsLoss`가 내부적으로 Sigmoid를 처리하고, 추론 시에는 `torch.sigmoid(output) > 0.5`로 이진 마스크를 생성한다.

---

## 5. 학습 설정

| 하이퍼파라미터 | 값 |
|---------------|-----|
| 입력 해상도 | 256 × 256 |
| Batch Size | 16 |
| Optimizer | Adam (lr=1e-4) |
| Loss Function | BCEWithLogitsLoss |
| Epochs | 10 |
| num_workers | 4 |
| pin_memory | True |

**BCEWithLogitsLoss** 선택 이유: Sigmoid + BCE를 log-sum-exp 트릭으로 수치적으로 안정하게 계산하며, 이진 세그멘테이션의 표준 손실 함수이다.

---

## 6. 평가 지표

`calculate_metrics()` 함수에서 두 지표를 동시에 계산한다.

### Dice Score

$$\text{Dice} = \frac{2 \cdot |P \cap G| + \varepsilon}{|P| + |G| + \varepsilon}$$

### IoU (Jaccard Index)

$$\text{IoU} = \frac{|P \cap G| + \varepsilon}{|P \cup G| + \varepsilon}$$

- `smooth = 1e-6 (ε)`: 분모가 0이 되는 수치 불안정 방지
- 예측값은 `sigmoid(output) > 0.5`로 이진화 후 계산
- **Validation IoU 기준**으로 Best Model 저장 (`best_unet_model.pth`)

배경이 차량보다 훨씬 많은 불균형 데이터 특성상, 픽셀 단순 정확도보다 영역 중첩 기반의 Dice·IoU가 적합한 지표이다.

---

## 7. 학습 루프

```
for epoch in range(10):

    # Train
    model.train()
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)   # BCEWithLogitsLoss
        loss.backward()
        optimizer.step()
        → Dice, IoU 누적

    # Validation
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            outputs = model(images)
            → Loss, Dice, IoU 누적

    # 에폭 평균 출력
    # Best Model 저장 (Val IoU 갱신 시)
    if avg_val_iou > best_val_iou:
        torch.save(model.state_dict(), "best_unet_model.pth")
```

---

## 8. 추론 및 시각화 (`visualize.py`)

1. `best_unet_model.pth` 로드 → `model.eval()`
2. 테스트 이미지 상위 5장 로드 → 전처리 (Resize, Normalize)
3. `unsqueeze(0)` 배치 차원 추가 → GPU 이동
4. `model(input)` → `sigmoid()` → `> 0.5` 임계화 → NumPy 변환
5. `matplotlib`으로 원본 이미지와 예측 마스크를 좌우 배치하여 `test_visualization.png` 저장

---

## 9. 구현 설계 요약

| 구성 요소 | 선택 및 근거 |
|----------|-------------|
| **모델** | U-Net — Skip Connection으로 경계 복원이 우수한 검증된 세그멘테이션 구조 |
| **손실 함수** | BCEWithLogitsLoss — 이진 분류 표준, 수치 안정성 확보 |
| **평가 지표** | Dice + IoU — 불균형 데이터에서 영역 기반 지표가 픽셀 정확도보다 적합 |
| **데이터 증강** | HorizontalFlip — 차량 이미지 특성상 좌우 반전이 자연스러운 증강 |
| **모델 저장** | Val IoU 기반 Best Checkpoint — 과적합 방지, 최적 가중치 보존 |

---

## 10. 파일 구조

| 파일 | 역할 |
|------|------|
| `dataset.py` | `CarvanaDataset` — 이미지·마스크 로드, 전처리 적용 |
| `model.py` | `DoubleConv`, `UNet` 구현 |
| `train.py` | 데이터 분할, 학습·검증 루프, 지표 계산, 모델 저장 |
| `visualize.py` | 학습된 모델로 테스트 추론 및 시각화 |
