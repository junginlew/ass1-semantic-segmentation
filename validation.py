# validation.py (평가 루프 예시)
outputs = model(images) # 여전히 날것의 로짓(-5.0 ~ 5.0)

# 1. 수동으로 Sigmoid 함수를 통과시켜 0.0 ~ 1.0 사이의 확률로 변환합니다.
probs = torch.sigmoid(outputs)

# 2. 임계값(Threshold) 설정: 확률이 50% 이상(0.5)이면 자동차(1), 아니면 배경(0)으로 픽셀을 확정 짓습니다.
predicted_masks = (probs > 0.5).float()

# 3. 이제 이 predicted_masks와 정답지(masks)를 비교해서 점수를 매기거나 그림을 그립니다!