import tensorrt as trt
import os

def check_tensorrt_version():
    print("TensorRT version: ", trt.__version__)

def build_engine(onnx_file_path, engine_file_path, max_workspace_size=1 << 30):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Set max workspace size and precision mode
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    config.set_flag(trt.BuilderFlag.FP16)  

    # Build serialized network
    print("Building TensorRT engine... This may take a few minutes.")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build the serialized network.")
        return None

    print("Engine built successfully, saving to file...")
    
    # Save the engine to file directly
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"Successfully saved TensorRT engine to {engine_file_path}")

    return True

if __name__ == "__main__":
    onnx_model_path = "../model.onnx"
    tensorrt_model_path = "model.plan"  # Using .plan extension to indicate TensorRT engine

    check_tensorrt_version()

    if not os.path.exists(onnx_model_path):
        print(f"ONNX model file {onnx_model_path} does not exist.")
    else:
        result = build_engine(onnx_model_path, tensorrt_model_path)
        if result:
            print("Conversion completed successfully!")
        else:
            print("Conversion failed!")
