import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # CUDA 디바이스 자동 초기화 및 컨텍스트 생성
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

ENGINE_FILE_PATH = "unet_model.engine"
TEST_IMAGE_PATH ="./data/test/0ab2e2edfa32_15.jpg"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def allocate_buffers(engine):
    inputs = []
    outputs = []
    stream = cuda.Stream()
    
    for tensor_name in engine:
        shape = engine.get_tensor_shape(tensor_name)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        
        host_mem = cuda.pagelocked_empty(size, dtype) # Host Memory 할당 (페이지 폴트 방지를 위해 pagelocked 사용)
        device_mem = cuda.mem_alloc(host_mem.nbytes) # Device Memory 할당 (GPU VRAM 공간 확보)
    
        # 입력/ 출력 모드 확인 후 저장
        mode = engine.get_tensor_mode(tensor_name)
        if mode == trt.TensorIOMode.INPUT:
            inputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem})
        elif mode == trt.TensorIOMode.OUTPUT:
            outputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem})
       
    return inputs, outputs, stream

if __name__ == '__main__':
    print("Loading TensorRT engine...")
    
    with open(ENGINE_FILE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())  # 직렬화된 바이너리를 읽어서 GPU 메모리에 실행 가능한 형태로 복원
    context = engine.create_execution_context()
    
    inputs, outputs, stream = allocate_buffers(engine) #입출력 버퍼, 스트림 할당
    print("Host(RAM), Device(GPU) buffers and CUDA stream allocated.")
    
    #데이터 전처리
    image_bgr = cv2.imread(TEST_IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    input_tensor = transform(image=image_rgb)["image"]
    input_numpy = input_tensor.unsqueeze(0).numpy().astype(np.float32)
    
    #TensorRT 추론
    print("Running inference...")
    for inp in inputs:
        context.set_tensor_address(inp['name'], int(inp['device']))
    for out in outputs:
        context.set_tensor_address(out['name'], int(out['device']))
    
    np.copyto(inputs[0]['host'], input_numpy.ravel()) #입력 텐서를 1차원으로 펼쳐서 Host의 핀 메모리 버퍼로 복사
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream) # HostToDevice (RAM에서 GPU VRAM으로)
    context.execute_async_v3(stream_handle=stream.handle) # GPU에서 추론 수행
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream) # DeviceToHost: 연산 결과를 VRAM -> RAM   
    stream.synchronize() # 스트림 동기화 (GPU 작업 완료까지 CPU 대기)
    
    output_logits = outputs[0]['host'].reshape((1, 1, 256, 256)) # 1차원 배열을 4차원으로 복원
    
    # Sigmoid 없이 output_logits > 0 이면 1, 아니면 0으로 이진 마스크 생성
    pred_mask_np = (output_logits > 0).astype(np.uint8)
    pred_mask_np = np.squeeze(pred_mask_np) # 2D 배열로 변환

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.resize(image_rgb, (256, 256)))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask_np, cmap='gray')
    plt.title("TensorRT (FP16 Optimized) Output")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("trt_inference_result.png")
    print("TensorRT inference result saved to 'trt_inference_result.png'")