FROM nvcr.io/nvidia/tritonserver:23.12-py3

USER root

# Copy and install Python requirements
COPY deployment/requirements.server.txt .

# Install requirements with proper dependency resolution
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.server.txt

# Install system dependencies for OpenMMLab and MMPose
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    build-essential && \
    rm -rf /var/lib/apt/lists/*


# Install numpy and the official MMLab installer tool 'mim'
RUN pip install --no-cache-dir "numpy==1.24.3"
RUN pip install --no-cache-dir openmim

# Use mim to install the full, compiled version of mmcv
RUN mim install "mmcv==2.1.0"

# Install the other compatible packages, allowing pip to handle dependencies
RUN pip install --no-cache-dir "mmengine==0.10.1"
RUN pip install --no-cache-dir "mmdet==3.2.0"
# Use mmpose==1.3.1 for compatibility with mmdet==3.2.0
RUN pip install --no-cache-dir "mmpose==1.3.1"
RUN pip install --no-cache-dir "xtcocotools"


# Download MMPose model weights (body keypoint detection)
RUN mkdir -p /models/mmpose_models && \
    cd /models/mmpose_models && \
    wget https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth


# Copy model repository (build context is now parent directory)
COPY models/model_repository /models

# Convert ONNX MLP model to TensorRT FP16 with correct input shapes (51 features)
RUN trtexec --onnx=/models/mlp_phone_detector/1/model.onnx \
            --saveEngine=/models/mlp_phone_detector/1/model.plan \
            --fp16 \
            --minShapes=input:1x51 \
            --optShapes=input:4x51 \
            --maxShapes=input:8x51 \
            --verbose

CMD ["tritonserver", "--model-repository=/models", "--allow-http=true", "--allow-grpc=true", "--allow-metrics=true", "--http-port=8000", "--grpc-port=8001", "--metrics-port=8002", "--log-verbose=1", "--exit-on-error=false", "--strict-model-config=false"]
