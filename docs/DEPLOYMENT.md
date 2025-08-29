# Triton Phone Detection - Deployment 

A neural network-based phone detection system using HRNet pose estimation and a Multi-Layer Perceptron (MLP) deployed on NVIDIA Triton Inference Server.


### Pipeline Overview
```
Image Input → Preprocessing (Python/MMPose) → MLP (ONNX) → Postprocessing (Python) → Results
                    ↓                          ↓              ↓
              Triton Server with Dynamic Batching, GPU Acceleration & Load Balancing
```

### Performance Metrics
- **ONNX Runtime**: ~0.08ms inference time (**10x faster** than PyTorch)
- **Accuracy**: 90.71% 
- **gRPC Protocol**: ~40% faster than HTTP
- **Model Size**: 0.06 MB

## 📁 Production Structure

```
production-phone-detection/
├── deployment/           # Production deployment files
│   ├── deploy.sh        # Automated deployment script
│   ├── docker-compose.yml
│   ├── Dockerfile.client
│   └── requirements.client.txt
├── models/              # Model artifacts
│   ├── mlp_phone_detector_best.pth
│   ├── mlp_phone_detector.onnx
│   └── model_repository/    # Triton model repository
├── clients/             # Client applications
│   ├── triton_client.py     # Main production client
│   ├── client_grpc.py       # gRPC-only client
├── training/           
│   ├── train_mlp.py
│   ├── data_collector.py
│   ├── export_onnx.py
│   └── requirements.training.txt
└── docs/               # Documentation
    ├── DEPLOYMENT_GUIDE.md 
    ├── GRPC_IMPLEMENTATION.md
    └── DEPLOYMENT_SUMMARY.md
```

## Quick Start (Production)

### 1. System Requirements
- **NVIDIA GPU**: Recommended for optimal performance
- **Docker**: For containerized deployment
- **Python 3.8+**: For client applications

### 2. Deploy Triton Server
```bash
cd deployment/
./deploy.sh
```

### 3. Test Deployment
```bash
# Health check (gRPC - default and faster)
python ../clients/triton_client.py --health-check

# Run inference on image
python ../clients/triton_client.py --image test_image.jpg

# Performance benchmark
python ../clients/triton_client.py --image test_image.jpg --benchmark 20

# Protocol comparison
python ../clients/test_protocol_performance.py --image test_image.jpg --requests 20
```

## Deployment Options

### Option A: Docker Compose (Recommended)
```bash
cd deployment/
docker-compose up triton-server
```

### Option B: Manual Triton Server
```bash
# Install Triton Server
docker pull nvcr.io/nvidia/tritonserver:24.01-py3

# Run server
docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 \
    -v$(pwd)/models/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models
```

## Client Usage

### gRPC Client (Recommended - Best Performance)
```bash
# Basic usage
python clients/triton_client.py --image path/to/image.jpg

# With benchmarking
python clients/triton_client.py --image image.jpg --benchmark 10

# gRPC-only client
python clients/client_grpc.py --image image.jpg --benchmark 20
```

### HTTP Client (Fallback)
```bash
# Force HTTP protocol
python clients/triton_client.py --image image.jpg --use-http

# Specific HTTP URL
python clients/triton_client.py --triton-url http://localhost:8000 --use-http
```

### API Response Format
```json
{
  "person_id": 0,
  "is_phone": true,
  "confidence": 0.9525
}
```

## Production Configuration

### Server Configuration
- **HTTP Port**: 8000 (fallback compatibility)
- **gRPC Port**: 8001 (recommended default)  
- **Metrics Port**: 8002 (monitoring)
- **Dynamic Batching**: Enabled for throughput
- **GPU Acceleration**: Auto-detected

### Model Repository Structure
```
model_repository/
├── preprocess/1/model.py          # Keypoint extraction (Python backend)
├── mlp_phone_detector/1/model.onnx # MLP inference (ONNX Runtime backend)  
├── postprocess/1/model.py         # Result formatting (Python backend)
└── ensemble_phone_detection/       # Pipeline orchestration
```

### Data Flow
1. **Image Input**: Raw image bytes via HTTP/gRPC
2. **Pose Estimation**: MMPose HRNet extracts 17 COCO keypoints
3. **Normalization**: Keypoints centered around neck, scaled by body size
4. **Feature Vector**: 51-element vector (x, y, confidence for each keypoint)
5. **ONNX Inference**: MLP with [128, 64] hidden layers
6. **Postprocessing**: Softmax, confidence thresholding (0.5), formatting
7. **Response**: JSON with person IDs, predictions, confidence scores


### Security
- **TLS Encryption**: Enable for production gRPC endpoints
- **Authentication**: Implement API key or token-based auth
- **Rate Limiting**: Configure request throttling
- **Network Security**: Use firewalls and VPN access

### Monitoring & Logging
```bash
# View server logs
docker logs triton-server

# Check model status
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models/ensemble_phone_detection
```

### Performance Tuning
- **Instance Groups**: Configure multiple model instances
- **Dynamic Batching**: Tune batch size and delay
- **Memory Optimization**: Set model memory pools
- **GPU Memory**: Monitor CUDA memory usage

## Troubleshooting

### Common Issues

**1. Server Won't Start**
```bash
# Check model repository structure
ls -la models/model_repository/

# Verify model configs
cat models/model_repository/*/config.pbtxt
```

**2. gRPC Connection Failed**
```bash
# Test with HTTP first
python clients/triton_client.py --image test.jpg --use-http

# Check if gRPC dependencies installed
pip list | grep tritonclient
```

**3. Model Loading Errors**
```bash
# Check Triton logs for detailed error messages
docker logs triton-server

# Verify ONNX model compatibility
python training/export_onnx.py
```

**4. Performance Issues**
```bash
# Check GPU availability
nvidia-smi

# Monitor resource usage
docker stats triton-server

# Run performance comparison
python clients/test_protocol_performance.py --image test.jpg
```

### Health Checks
```bash
# Server health
curl http://localhost:8000/v2/health/live
curl http://localhost:8000/v2/health/ready

# Model health
python clients/triton_client.py --health-check

# End-to-end test
python clients/triton_client.py --image test_image.jpg
```


### Model Optimization
- **TensorRT**: Convert ONNX to TensorRT for NVIDIA GPUs
- **Batch Processing**: Implement batch inference endpoints

## Model Updates

### Deploying New Models
```bash
# 1. Export new ONNX model
cd training/
python export_onnx.py

# 2. Update model repository
cp new_model.onnx ../models/model_repository/mlp_phone_detector/2/model.onnx

# 3. Update config for new version
vim ../models/model_repository/mlp_phone_detector/config.pbtxt

# 4. Restart server (or use model management API)
./deployment/deploy.sh
```

### Model Validation
```bash
# Test new model version
python clients/triton_client.py --image test.jpg
```

# gRPC Implementation for Triton Phone Detection

## Implementation Summary

Successfully upgraded your Triton phone detection system to use **gRPC as the primary protocol** for client communication, providing significant performance improvements over HTTP.

## What Was Implemented

### 1. **gRPC Client Integration**
- **client_test.py**: Updated to support both gRPC and HTTP protocols
- **client_grpc.py**: Dedicated gRPC-only client for maximum performance
- **Automatic fallback**: Falls back to HTTP if gRPC is unavailable

### 2. **Protocol Performance Optimization**
- **Default Protocol**: gRPC (port 8001) instead of HTTP (port 8000)
- **Binary Protocol**: More efficient than JSON for large payloads
- **Streaming Support**: Ready for future real-time enhancements

### 3. **Comprehensive Testing Suite**
- **Protocol Comparison**: `test_protocol_performance.py` benchmarks both protocols
- **Health Checks**: gRPC-native server and model readiness checks
- **Performance Benchmarks**: Enhanced metrics with percentiles and throughput


##  Usage Examples

### gRPC Client (Default)
```bash
# Health check via gRPC
python client_test.py --health-check

# Single image inference via gRPC  
python client_test.py --image test.jpg

# Performance benchmark via gRPC
python client_test.py --image test.jpg --benchmark 20

# Dedicated gRPC client
python client_grpc.py --image test.jpg --benchmark 20
```

### HTTP Fallback
```bash
# Force HTTP usage
python client_test.py --image test.jpg --use-http

# HTTP-specific URL
python client_test.py --triton-url http://localhost:8000 --use-http
```

### Protocol Comparison
```bash
# Compare gRPC vs HTTP performance
python test_protocol_performance.py --image test.jpg --requests 20
```

##  Architecture Changes

### Server Configuration
- **Triton Server**: Both HTTP (8000) and gRPC (8001) endpoints enabled
- **No server changes**: Existing deployment works with both protocols
- **Load balancing**: gRPC supports more efficient load balancing

### Client Architecture
```python
# Automatic protocol selection
use_grpc = not args.use_http and GRPC_AVAILABLE

# Dynamic response parsing
is_grpc_response = use_grpc and hasattr(response, 'as_numpy')
results = parse_triton_response(response, is_grpc_response)
```

### Immediate Use:
1. **Deploy server**: `./deploy.sh` (enables both protocols)
2. **Test gRPC**: `python client_test.py --health-check`
3. **Benchmark**: `python test_protocol_performance.py --image test.jpg`
