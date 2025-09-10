# Phone Detection CNN with Triton Inference Server

Human Pose Estimation + Multi Layer Perceptron deployed on Triton.

## Quick Start

```bash
# Deploy server
cd deployment && ./deploy.sh

# Test with production client
pip install tritonclient[grpc] numpy opencv-python
python predict.py --url localhost:8011 --image path/to/image.jpg
```

## Production Usage

### Python Integration

```python
from predict import PhoneDetectionClient

# Initialize client
detector = PhoneDetectionClient("your-server:8001") 

# Single image detection
result = detector.detect_phone_usage("image.jpg")
print(result)
# {"person_id": 0, "is_phone": true, "confidence": 0.85, "inference_time_ms": 340.2, "status": "success"}

# Batch processing
results = detector.detect_batch(["img1.jpg", "img2.jpg", "img3.jpg"])

# Health check
health = detector.health_check()
```

### Command Line Usage

```bash
# Health check
python predict.py --url localhost:8011 --health

# Single image
python predict.py --url localhost:8011 --image photo.jpg

# Batch processing
python predict.py --url localhost:8011 --batch img1.jpg img2.jpg img3.jpg

# Model information
python predict.py --url localhost:8011 --info
```

## Table of Contents

- [System Architecture](#system-architecture)
- [Requirements](#requirements)
- [Directory Structure](#directory-structure)
- [Deployment Methods](#deployment-methods)
  - [Method 1: Automated Deployment (deploy.sh)](#method-1-automated-deployment-deploysh)
  - [Method 2: Manual Docker Compose](#method-2-manual-docker-compose)
  - [Method 3: Manual Docker Commands](#method-3-manual-docker-commands)
- [Testing and Usage](#testing-and-usage)
- [deploy.sh Deep Dive](#deploysh-deep-dive)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Performance Metrics](#performance-metrics)
- [Development Guide](#development-guide)

## System Architecture

### Components Overview

```
Image Input → Preprocessing → MLP Classification → Postprocessing → Results
     ↓             ↓               ↓                 ↓
  MMPoseInferencer Keypoint      ONNX Runtime     JSON Response
      (CPU)      Normalization      (CPU)         (gRPC/HTTP)
     ↓             ↓               ↓                 ↓
         Triton Server with Dynamic Batching (CPU Optimized)
```

**1. Preprocessing Module (Python Backend)**
- MMPoseInferencer with RTMPose models for human pose estimation
- Extracts 17 COCO keypoints from human subjects using RTMDet person detection
- Normalizes keypoints relative to body scale and position
- Generates 51-dimensional feature vector

**2. MLP Classification Model (ONNX)**
- Multi-Layer Perceptron: [51] → [128] → [64] → [2]
- ReLU activation with 0.2 dropout
- Binary classification (phone/no-phone)
- CPU inference via ONNX Runtime

**3. Postprocessing Module (Python Backend)**
- Applies softmax to model logits
- Confidence thresholding (0.5)
- Formats structured JSON responses

**4. Ensemble Pipeline**
- Triton ensemble model orchestrating the pipeline
- Automatic batching and parallel processing
- Multiple protocol support (gRPC/HTTP)

## Requirements

### Dependencies
- Docker 24.0+
- Python 3.8+ (for client)
- System RAM: 4GB+ recommended

### Optional (for GPU optimization)
- NVIDIA Container Toolkit
- CUDA 12.8+ drivers
- GPU with 4GB+ VRAM

## Directory Structure

```
phone-detection-cnn-master/
├── deployment/                    # Deployment configurations
│   ├── deploy.sh                 # Automated deployment script
│   ├── docker-compose.yml        # Docker Compose configuration
│   ├── Dockerfile                # Custom Triton container
│   ├── requirements.server.txt   # Server Python dependencies
│   ├── client_grpc.py            # gRPC test client
├── models/model_repository/       # Triton model repository
│   ├── preprocess/               # MMPose preprocessing model
│   │   ├── config.pbtxt          # Model configuration
│   │   └── 1/model.py            # Python backend implementation
│   ├── mlp_phone_detector/       # ONNX classification model
│   │   ├── config.pbtxt          # Model configuration
│   │   └── 1/model.onnx          # ONNX model weights
│   ├── postprocess/              # Result postprocessing
│   │   ├── config.pbtxt          # Model configuration
│   │   └── 1/model.py            # Python backend implementation
│   └── ensemble_phone_detection/ # Pipeline orchestration
│       ├── config.pbtxt          # Ensemble configuration
│       └── 1/                    # Version directory (empty)
├── training/                      # Model training scripts
│   ├── train_mlp.py              # Training pipeline
│   ├── export_onnx.py            # ONNX conversion
│   ├── data_collector.py         # Dataset preparation
│   └── eval.py                   # Model evaluation
```

## Deployment Methods

### Method 1: Automated Deployment (deploy.sh)

The `deploy.sh` script provides one-command deployment with comprehensive error handling and health checks.

```bash
cd deployment

# Deploy and start (default)
./deploy.sh

# Other commands
./deploy.sh stop          # Stop the server
./deploy.sh restart       # Restart the server
./deploy.sh logs          # View logs
./deploy.sh status        # Check server status
./deploy.sh clean         # Remove containers and images
./deploy.sh help          # Show help
```

**What deploy.sh does:**
1. **Pre-flight checks**: Verifies Docker, GPU, model repository
2. **Image building**: Creates custom Triton image with dependencies
3. **Container management**: Stops/removes existing containers
4. **Server startup**: Launches with optimal GPU configuration
5. **Health monitoring**: Waits for all models to load successfully
6. **Validation**: Tests all model endpoints

### Method 2: Manual Docker Compose

For users who prefer manual control or want to understand the deployment process:

```bash
cd deployment

# Build the custom image
docker compose build

# Start services in background
docker compose up -d

# View logs (follow mode)
docker compose logs -f

# Check container status
docker compose ps

# Stop services
docker compose down

# Stop and remove images
docker compose down --rmi all
```

**Configuration customization:**
Edit `docker-compose.yml` to modify:
- Port mappings (default: 8000, 8001, 8002)
- GPU allocation
- Volume mounts
- Environment variables


## Testing and Usage

### Health Check

```bash
cd deployment

# Using the gRPC client
/usr/bin/python3 client_grpc.py --health-check

# Using curl (HTTP endpoint)
curl http://localhost:8000/v2/health/ready
```

### Basic Inference

```bash
cd deployment

# Test with a single image
/usr/bin/python3 client_grpc.py --image path/to/image.jpg

# With verbose output
/usr/bin/python3 client_grpc.py --image path/to/image.jpg --verbose
```

### Performance Benchmarking

```bash
cd deployment

# Run 50 requests benchmark
/usr/bin/python3 client_grpc.py --image path/to/image.jpg --benchmark 50

# Compare gRPC vs HTTP performance
/usr/bin/python3 client_grpc.py --image path/to/image.jpg --compare
```

### Installing Client Dependencies

```bash
# Install required packages for system Python
/usr/bin/python3 -m pip install --user numpy opencv-python tritonclient[grpc]
```

### Configuration Variables

```bash
MODEL_REPO_DIR="../models/model_repository"    # Model repository path
CUSTOM_TRITON_IMAGE="phone-detection-server:latest"  # Image name
CONTAINER_NAME="phone-detection"               # Container name
HTTP_PORT=8000                                # HTTP endpoint port
GRPC_PORT=8001                                # gRPC endpoint port
METRICS_PORT=8002                             # Metrics endpoint port
```

## API Documentation

### Input Format

**Image Input** (INPUT_IMAGE):
- **Type**: UINT8 array
- **Format**: Encoded image bytes (JPEG/PNG)
- **Size**: Variable length byte array
- **Batch**: Support for batch sizes 1-4

### Output Format

**Response Structure**:
```json
{
  "person_id": 0,        // Person identifier (int32)
  "is_phone": true,      // Phone detection result (boolean)
  "confidence": 0.8547   // Confidence score 0.0-1.0 (float32)
}
```

### HTTP REST API

```bash
# Health check
curl http://localhost:8000/v2/health/ready

# Model status
curl http://localhost:8000/v2/models/ensemble_phone_detection

# Inference (JSON payload required)
curl -X POST http://localhost:8000/v2/models/ensemble_phone_detection/infer \
     -H "Content-Type: application/json" \
     -d @inference_request.json
```

## Performance Metrics

### Model Performance
- **Accuracy**: 90.71% on validation dataset
- **Model Size**: 0.06 MB (ONNX optimized)
- **Input Processing**: 17 keypoints per person, batch sizes up to 4
- **Feature Extraction**: 51-dimensional normalized pose vectors

### System Performance (Current CPU Setup)
- **gRPC Latency**: 350-570ms average (CPU MMPose + CPU MLP)
- **HTTP Latency**: Similar to gRPC for CPU workloads
- **Throughput**: 2-3 requests/second (CPU optimized)
- **CPU Utilization**: 70-90% during inference
- **Memory Usage**: ~1GB System RAM

### Benchmark Results (CPU Setup)
```
gRPC Performance Results:
   Successful requests: 10/10
   Average latency: 410.29 ms
   Min latency: 333.15 ms
   Max latency: 577.47 ms
   95th percentile: 464.15 ms
   99th percentile: 565.13 ms
   Throughput: 2.36 requests/second
```

### Performance Notes
- Current system runs entirely on CPU for maximum stability
- MMPose RTMPose inference: ~300ms (majority of latency)
- MLP classification: ~10ms
- GPU optimization planned for significant speedup (3-5x improvement expected)


### Configuration Customization

**Triton Model Config** (`config.pbtxt`):
- Batch size limits
- Input/output tensor specifications
- Backend parameters
- Optimization settings

**Ensemble Pipeline** (`ensemble_phone_detection/config.pbtxt`):
- Model sequencing
- Input/output mapping
- Scheduling parameters


### Working Features
- ✅ Human pose detection using RTMPose (RTMDet + RTMPose-m)
- ✅ Phone usage classification with realistic confidence scores
- ✅ Batch processing support (1-4 images)
- ✅ gRPC and HTTP API endpoints
- ✅ Health monitoring and status checks
- ✅ Automated deployment with error handling

### Known Limitations
- **Latency**: ~400ms inference time (CPU bottleneck)
- **Throughput**: ~2-3 requests/second (suitable for batch processing)
- **GPU Support**: Currently disabled (planned optimization)

## Planned Enhancements

### Next Phase: GPU Optimization
1. **TensorRT FP16 MLP**: Convert ONNX model to TensorRT for 3-5x speedup
2. **GPU MMPose**: Move pose estimation to GPU for additional performance gains
3. **Latency Target**: Reduce from ~400ms to ~100ms total inference time
