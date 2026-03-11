import torch
from model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("best_unet_model.pth", weights_only=True)) # 학습된 가중치 불러오기 , weights_only=True로 보안 강화
model.eval() #평가모드로 설정

dummy_input = torch.randn(1, 3, 256, 256, device=device) # randn: 평균 0, 표준편차 1의 가우시안 표준정규분포로 값을 채움
onnx_file_path = "unet_model.onnx"

torch.onnx.export(
    model,
    dummy_input,            # 모델의 입력 규격을 알려주기 위함
    onnx_file_path,
    export_params=True,     # 모델의 가중치 저장
    opset_version=11,       # ONNX 버전
    do_constant_folding=True, # 상수 폴딩 최적화 (불필요한 연산을 컴파일 타임에 미리 계산)
    input_names=['input'],  # 입력 노드 이름
    output_names=['output'] # 출력 노드 이름
)
print(f"Model has been exported to {onnx_file_path}")