# Phone Detection CNN with Triton Inference Server

Human Pose Estimation + Multi Layer Perceptron deployed on Triton.


```bash
# Deploy server
cd deployment && ./deploy.sh

# Test with production client
pip install tritonclient[grpc] numpy opencv-python
python predict.py --url localhost:8011 --image path/to/image.jpg
```


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

## Results
```bash
cv-laptop-1@cv-laptop-1:~/Desktop/samit/triton/phone-detection-cnn-master/deployment$ /usr/bin/python3 predict.py --url localhost:8011 --image test_images/phone.png
INFO:__main__:Successfully connected to Triton server at localhost:8011

Detection Result: {"person_id": 0, "is_phone": True, "confidence": 0.6212334036827087, "inference_time_ms": 414.24, "status": "success"}

cv-laptop-1@cv-laptop-1:~/Desktop/samit/triton/phone-detection-cnn-master/deployment$ /usr/bin/python3 predict.py --url localhost:8011 --image test_images/nophone3
INFO:__main__:Successfully connected to Triton server at localhost:8011

Detection Result: {"person_id": 0, "is_phone": False, "confidence": 0.4736001789569855, "inference_time_ms": 302.46, "status": "success"}
```

## System Architecture

### Components Overview

```
Image Input → preprocess_img → rtmdet_detection → person_cropper → rtmpose_estimation → feature_normalizer → mlp_phone_detector → postprocess → Results
     ↓              ↓                 ↓                 ↓                  ↓                     ↓                     ↓                  ↓
  Raw Image     Image Resize/     Person BBox       Pose Keypoints     Normalized Pose       Phone/No-Phone        Formatted JSON     Final
  (UINT8)       Normalization     (x1,y1,x2,y2)     (simcc_x, simcc_y)  Features (51-dim)     Logits (2-dim)        Response           Output
     ↓              ↓                 ↓                 ↓                  ↓                     ↓                     ↓                  ↓
                                                                Triton Inference Server Ensemble
```

**1. `preprocess_img` (Python Backend)**
- Resizes and normalizes input images (e.g., to 640x640) for the RTMDet model.
- Converts image format (BGR to RGB) and transposes dimensions.

**2. `rtmdet_detection` (ONNX Model)**
- Performs object detection (likely for persons) on the preprocessed image.
- Outputs bounding box detections (`dets`) and corresponding labels (`labels`).

**3. `person_cropper` (Python Backend)**
- Takes the original preprocessed image and the detections from `rtmdet_detection`.
- Crops the image to focus on the detected person (assuming person class 0).
- Resizes the cropped person image to a standard size (e.g., 192x256) for pose estimation.

**4. `rtmpose_estimation` (TensorRT/ONNX Model)**
- Estimates 17 COCO keypoints for the cropped person image.
- Outputs pose estimation results in SIMCC format (`simcc_x`, `simcc_y`).

**5. `feature_normalizer` (Python Backend)**
- Converts SIMCC outputs from `rtmpose_estimation` into normalized (x,y) keypoint coordinates.
- Extracts additional geometric features (e.g., body width, height, aspect ratio) from the keypoints.
- Generates a 51-dimensional feature vector for the MLP classifier.

**6. `mlp_phone_detector` (ONNX Model)**
- A Multi-Layer Perceptron (MLP) model.
- Takes the 51-dimensional normalized pose features as input.
- Performs binary classification (phone/no-phone) based on the pose.
- CPU inference via ONNX Runtime.

**7. `postprocess` (Python Backend)**
- Applies softmax to the raw logits from `mlp_phone_detector` to get probabilities.
- Applies a confidence threshold (0.5) to determine the final classification.
- Formats the results into a structured JSON response, including `person_id`, `is_phone`, and `confidence`.

**8. Ensemble Pipeline**
- The entire process is orchestrated by a Triton ensemble model (`ensemble_phone_detection`).
- Manages the sequential execution and data flow between all the individual models.
- Supports automatic batching and parallel processing.
- Provides multiple protocol support (gRPC/HTTP) for client interaction.


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
- Human pose detection using RTMPose (RTMDet + RTMPose-m)
- Phone usage classification with realistic confidence scores
- Batch processing support (1-4 images)
- gRPC and HTTP API endpoints
- Health monitoring and status checks
- Automated deployment with error handling

### Known Limitations
- **Latency**: ~400ms inference time (CPU bottleneck)
- **Throughput**: ~2-3 requests/second (suitable for batch processing)
- **GPU Support**: Currently disabled (planned optimization)

## Planned Enhancements

### Next Phase: GPU Optimization
1. **TensorRT FP16 MLP**: Convert ONNX model to TensorRT for 3-5x speedup
2. **GPU MMPose**: Move pose estimation to GPU for additional performance gains
3. **Latency Target**: Reduce from ~400ms to ~100ms total inference time
