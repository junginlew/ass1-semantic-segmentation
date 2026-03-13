import torch
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from model import UNet

BATCH_SIZE = 1
IMG_H, IMG_W = 256, 256
NUM_WARMUP = 50      # GPU 예열 횟수
NUM_ITERATIONS = 500 # 실제 측정 횟수

dummy_input_np = np.random.randn(BATCH_SIZE, 3, IMG_H, IMG_W).astype(np.float32)

print("Starting benchmark...")
print(f"Warm Up: {NUM_WARMUP} iterations, Measurement: {NUM_ITERATIONS} iterations")

# Pytorch 모델 로드, 추론 함수
print("Loading PyTorch model...")
model_pt = UNet(in_channels=3, out_channels=1).cuda().eval()
model_pt.load_state_dict(load_file("best_unet_model.safetensors"))

def infer_pytorch(): #공정한 비교를 위해 텐서 변환, htod, dtoh 데이터 전송, numpy 변환 포함
    with torch.no_grad():
        gpu_input = torch.from_numpy(dummy_input_np).cuda()
        gpu_output = model_pt(gpu_input)
        _ = gpu_output.cpu().numpy()
    torch.cuda.synchronize() # .cpu().numpy() 가 이미 동기화를 유발하지만, 벤치마크의 정석을 위해 명시적 호출
    
# ONNX Runtime 세션 생성, 추론 함수
print("Loading ONNX model...")
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ort_session = ort.InferenceSession("unet_model.onnx", providers=providers)
input_name = ort_session.get_inputs()[0].name

def infer_onnx():
    # Numpy 배열을 입력-> C++ OrtValue(텐서)로 래핑-> HtoD -> 추론 -> DtoH 후 Numpy 반환
    _ = ort_session.run(None, {input_name: dummy_input_np})
    
#TensorRT 엔진 로드, 추론 함수
print("Loading TensorRT engine...")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("unet_model.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# VRAM, 핀 메모리 할당
inputs, outputs = [], []
stream = cuda.Stream()
for tensor_name in engine:
    shape = engine.get_tensor_shape(tensor_name)
    dtype= trt.nptype(engine.get_tensor_dtype(tensor_name))
    
    host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    mode = engine.get_tensor_mode(tensor_name)
    
    if mode == trt.TensorIOMode.INPUT:
        inputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem})       
    elif mode == trt.TensorIOMode.OUTPUT:
        outputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem})
    context.set_tensor_address(tensor_name, int(device_mem))
    
def infer_tensorrt():
    np.copyto(inputs[0]['host'], dummy_input_np.ravel())
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()

#벤치마킹 실행, 시간 측정
def run_benchmark(target_func, name):
    print(f"Warming up {name}...")
    for _ in range(NUM_WARMUP):  # 초기 지연, 하드웨어 가속 최적화 과정을 제외하기 위함
        target_func()
    print(f"Measuring {name} for {NUM_ITERATIONS} iterations...")
    start_time = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        target_func()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_latency_ms = (total_time / NUM_ITERATIONS) * 1000
    fps = 1000 / avg_latency_ms
    print(f"[{name}] 지연시간: {avg_latency_ms:.2f} ms | FPS: {fps:.2f} 프레임/초\n")
    return avg_latency_ms

#결과 시각화, 저장
print("-" * 50)
time_pt = run_benchmark(infer_pytorch, "PyTorch (FP32)")
time_ort = run_benchmark(infer_onnx, "ONNXRuntime (CUDA)")
time_trt = run_benchmark(infer_tensorrt, "TensorRT (FP16)")
print("-" * 50)

labels = ['PyTorch', 'ONNXRuntime', 'TensorRT (FP16)']
times = [time_pt, time_ort, time_trt]
colors = ['#EE4C2C', '#005BBB', '#76B900'] 

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, times, color=colors)
plt.ylabel('Average Latency (ms)')
plt.title('End-to-End Inference Time Comparison (RTX A6000)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (max(times)*0.02), f"{yval:.2f} ms", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("benchmark_result.png", dpi=300)
print("Benchmark result graph saved as 'benchmark_result.png'!")