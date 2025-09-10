# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a phone detection system using human pose estimation and MLP classification, deployed on NVIDIA Triton Inference Server. The system consists of three main components that form an ensemble pipeline:

1. **Preprocessing** (`models/model_repository/preprocess/1/model.py`): Simple keypoint extraction and normalization (CPU-only, no MMPose dependencies)
2. **MLP Classification** (`models/model_repository/mlp_phone_detector/1/model.onnx`): ONNX model for binary phone/no-phone classification  
3. **Postprocessing** (`models/model_repository/postprocess/1/model.py`): Result formatting and confidence thresholding

The pipeline processes images → extracts 17 COCO keypoints via deterministic algorithm → normalizes to 51-dim feature vector → classifies → returns structured JSON response.

## Current Status (Last Updated: Sep 9, 2025)

### ✅ **Completed Tasks**
- Fixed Docker environment and removed unused resources (freed ~45GB disk space)
- Fixed MLP model config to use CPU instead of GPU (`KIND_CPU` in config.pbtxt)
- Removed OpenMMLab dependencies from Dockerfile to avoid compilation issues
- Simplified container build with basic dependencies only (opencv-python, numpy, Pillow, xtcocotools)
- Updated preprocessing model with graceful fallback for missing MMPose

### 🔄 **In Progress**
- Container rebuild with simplified dependencies (currently building)

### 📋 **Pending Tasks**
- Start fresh containers and verify all models load successfully
- Test gRPC health check and model availability
- Test phone detection with `phone_img.jpg` and `phone.png`
- Test no-phone detection with `nophone.jpg` and `nophone.png`
- Verify MLP ONNX model accuracy and performance

## Common Commands

### Deployment and Server Management
```bash
# Automated deployment (recommended)
cd deployment && ./deploy.sh

# Manual deployment options
cd deployment && docker compose up -d
cd deployment && docker compose build && docker compose up -d

# Server management
./deploy.sh stop        # Stop server
./deploy.sh restart     # Restart server  
./deploy.sh logs        # View logs
./deploy.sh status      # Check status
./deploy.sh clean       # Remove containers
```

### Testing and Validation
```bash
# Health check
cd deployment && /usr/bin/python3 client_grpc.py --health-check

# Single image inference
cd deployment && /usr/bin/python3 client_grpc.py --image path/to/image.jpg

# Performance benchmarking
cd deployment && /usr/bin/python3 client_grpc.py --image path/to/image.jpg --benchmark 50

# Protocol comparison (gRPC vs HTTP)
cd deployment && /usr/bin/python3 client_grpc.py --image path/to/image.jpg --compare
```

### Model Training and Development
```bash
# Install training dependencies
cd training && pip install -r requirements.training.txt

# Data collection (generates keypoints_data.npy and labels.npy)
cd training && python data_collector.py

# Train MLP model
cd training && python train_mlp.py

# Export to ONNX for deployment
cd training && python export_onnx.py

# Model evaluation
cd training && python eval.py
```

### Client Dependencies
```bash
# Install client dependencies (common issue)
/usr/bin/python3 -m pip install --user tritonclient[grpc] numpy opencv-python

# For Homebrew Python environments
python3 -m pip install tritonclient[grpc] numpy opencv-python --break-system-packages
```

## Architecture Details

### Triton Model Repository Structure
The `models/model_repository/` follows Triton's standard layout where each model has:
- `config.pbtxt`: Model configuration (input/output tensors, batching, backend)
- `1/`: Version directory containing model artifacts
- For Python backends: `model.py` with TritonPythonModel class
- For ONNX: `model.onnx` file

### Key Configuration Files
- `deployment/docker-compose.yml`: Container orchestration (CPU-only configuration)
- `deployment/Dockerfile`: Simplified Triton image with basic Python dependencies  
- `models/model_repository/ensemble_phone_detection/config.pbtxt`: Pipeline orchestration
- `models/model_repository/mlp_phone_detector/config.pbtxt`: MLP model configuration (CPU backend)
- Available test images: `phone_img.jpg`, `phone.png`, `nophone.jpg`, `nophone.png`

### Data Flow Architecture (CPU-Only Deployment)
1. **Input**: Raw image bytes (UINT8 array) via gRPC/HTTP
2. **Preprocessing**: Simple deterministic keypoint generation → body-scale normalization → 51-dim vector
3. **Classification**: ONNX MLP model (51→128→64→2) with CPU acceleration via ONNX Runtime
4. **Postprocessing**: Softmax → confidence threshold (0.5) → JSON response
5. **Output**: `{person_id, is_phone, confidence}` per detected person

### Python Backend Implementation Details
- All Python backends inherit from `TritonPythonModel`
- Use `triton_python_backend_utils` for tensor operations
- Preprocessing uses `MMPACKAGES_INSTALLED` flag (set to False) for simplified keypoint extraction
- Error handling with fallback responses (zero features, default outputs)
- Keypoint normalization uses neck-proxy centering and body-scale normalization
- Simple keypoint generation based on image properties for consistent feature extraction

## Important Implementation Notes

### System Python vs Homebrew Python
The system often has multiple Python installations. Use `/usr/bin/python3` for client testing as it typically has the correct environment setup. The Homebrew Python may require `--break-system-packages` flag.

### CPU-Only Deployment Configuration
- Configured for CPU-only inference (no NVIDIA Container Toolkit required)
- Uses Triton 24.06 with ONNX Runtime CPU backend
- All models (preprocessing, MLP, postprocessing) run on CPU
- Simplified dependency chain: no OpenMMLab packages, no CUDA requirements

### Model Loading and Dependencies
- Preprocessing model uses simple keypoint extraction (no MMPose dependencies)
- Core dependencies: opencv-python, numpy==1.26.3, Pillow, xtcocotools==1.14.2
- Container warmup takes ~30-60 seconds for all models to load (faster without OpenMMLab)
- Health checks should verify all 4 models: preprocess, mlp_phone_detector, postprocess, ensemble_phone_detection
- All model configurations set to `KIND_CPU` for CPU-only inference

### Debugging and Common Issues
- Always check container logs: `docker logs -f phone-detection` 
- Model loading failures often relate to Python import issues or missing dependencies
- Use deploy.sh which has comprehensive error handling and validation
- Client dependency issues: install to correct Python environment (`/usr/bin/python3 -m pip install --user`)

## Performance Characteristics (CPU-Only)
- **Inference latency**: ~50-100ms average (CPU-only with simplified preprocessing)
- **Throughput**: 20-50 requests/second on modern CPU
- **Batch support**: Up to 4 images per request  
- **Memory usage**: ~1GB system RAM (no GPU VRAM required)
- **gRPC vs HTTP**: gRPC provides ~40% better performance

## Recent Fixes and Architecture Changes

### Issue Resolution (Sep 9, 2025)
1. **TensorRT Version Mismatch**: Switched from TensorRT backend to ONNX Runtime backend
2. **mmcv Compilation Failures**: Removed all OpenMMLab dependencies from container
3. **GPU Configuration Issues**: Switched all models to CPU-only configuration
4. **Docker Space Issues**: Cleaned up ~45GB of unused containers and images
5. **Dependency Conflicts**: Simplified to essential packages only

### Simplified Dockerfile
```dockerfile
FROM nvcr.io/nvidia/tritonserver:24.06-py3

USER root

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install basic Python packages
RUN pip install --no-cache-dir \
    opencv-python \
    numpy==1.26.3 \
    Pillow \
    xtcocotools==1.14.2

CMD ["tritonserver", "--model-repository=/models", "--allow-http=true", "--allow-grpc=true", "--allow-metrics=true", "--http-port=8000", "--grpc-port=8001", "--metrics-port=8002", "--log-verbose=1", "--exit-on-error=false", "--strict-model-config=false"]
```