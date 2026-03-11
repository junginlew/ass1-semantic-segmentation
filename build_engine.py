import tensorrt as trt
import os

ONNX_FILE_PATH = "unet_model.onnx"
ENGINE_FILE_PATH = "unet_model.engine"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING) #Warning 이상만 로그 출력

def build_engine(onnx_file_path, engine_file_path):
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) #모델의 batch 차원 명시적으로 정의하겠다는 flag
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    #ONNX 모델 파일 읽기, parsing
    if not os.path.exists(onnx_file_path):
        print(f"Error: {onnx_file_path} does not exist.")
        return
    
    with open(onnx_file_path, "rb") as model: #read binary
        if not parser.parse(model.read()):
            print("Failed to parse ONNX model:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return
        
    #Builder config 설정
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1 << 30)) #최적화 시뮬레이션을 위한 workspace 메모리 2GB
    
    # GPU의 FP16 고속 연산 지원 여부 확인
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 (Half Precision) mode is enabled.")
    else:
        print("Current hardware does not support fast FP16. Building with FP32.")
        
    #엔진 빌드
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Engine build failed.")
        return
    
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Engine has been built and saved to {engine_file_path}")
    
if __name__ == "__main__":
    build_engine(ONNX_FILE_PATH, ENGINE_FILE_PATH)