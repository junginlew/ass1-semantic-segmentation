import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import onnxruntime as ort

ONNX_MODEL_PATH = "unet_model.onnx"
TEST_IMAGE_PATH = "./data/test/0a84dac5df69_16.jpg"

if __name__ == '__main__':
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    #데이터 전처리
    image_bgr = cv2.imread(TEST_IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2(), #ONNX는 Tensor를 못읽지만 차원 순서 재배치를 위해 사용 (HWC → CHW)
    ])
    
    input_tensor = transform(image=image_rgb)["image"]
    input_numpy = input_tensor.unsqueeze(0).numpy() #Batch 차원 추가(1, 3, 256, 256), Numpy 배열로 변환 
    
    # ONNXRuntime으로 추론
    ort_outputs = ort_session.run(None, {input_name: input_numpy}) #None: 모든 출력 노드 가져옴
    output_logits = ort_outputs[0] #출력은 리스트로 나오므로 첫번째 요소 가져옴. (1, 1, 256, 256) numpy 배열
    
    ##  원래 Sigmoid 함수를 거치고 0.5를 기준으로 이진 마스크를 만들어야하지만, logit x >0 이면 Sigmoid(x) > 0.5 이고 logit x <0 이면 Sigmoid(x) <0.5 이므로 logit값을 기준으로 마스크 생성
    pred_mask_np = (output_logits > 0).astype(np.uint8)
    pred_mask_np = np.squeeze(pred_mask_np) #차원 축소 (256, 256)
    
    #결과 시각화
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.resize(image_rgb, (256, 256)))
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask_np, cmap='gray')
    plt.title("ONNXRuntime (CUDA) Output")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("onnx_inference_result.png")
    print("Inference completed and result saved as 'onnx_inference_result.png'")