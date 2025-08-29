### System Components

**1. Pose Estimation Module**
- HRNet model via MMPose framework
- Extracts 17 COCO keypoints from human subjects
- Pre-trained on COCO dataset for robust pose detection

**2. Feature Processing Pipeline**
- Keypoint normalization relative to body scale and position
- 51-dimensional feature vector (x, y, confidence per keypoint)
- Body-centered coordinate system using neck proxy

**3. Classification Model**
- Multi-Layer Perceptron (MLP) architecture
- Hidden layers: [128, 64] neurons with ReLU activation
- 0.2 dropout rate for regularization
- Binary classification output (phone/no-phone)

**4. Inference Server**
- NVIDIA Triton Inference Server for production deployment
- ONNX Runtime backend for optimized execution
- Dynamic batching and GPU acceleration
- Dual protocol support (gRPC/HTTP)

## Data Flow
##  Architecture

```
Image → Preprocessing (MMPose) → MLP (ONNX) → Postprocessing → Results
           ↓                      ↓              ↓
    Triton Server with Dynamic Batching & GPU Acceleration
```

```
Image Input → Pose Detection → Keypoint Extraction → Normalization → MLP Classification → Result Output
```

**Detailed Process:**
1. Raw image processing through MMPose HRNet
2. 17 keypoint extraction with confidence scores
3. Coordinate normalization using neck-centered scaling
4. Feature vector construction (51 elements)
5. MLP inference via ONNX Runtime
6. Softmax activation with 0.5 confidence threshold
7. JSON response with person ID, prediction, and confidence

## Performance Metrics

### Model Performance
- **Accuracy**: 90.71% on validation dataset
- **Inference Speed**: 0.08ms (ONNX) vs 0.83ms (PyTorch) - 10x improvement
- **Model Size**: 0.06 MB (highly efficient)
- **Input Processing**: 17 keypoints per person, batch sizes up to 8

### System Performance
- **Protocol Efficiency**: gRPC 40% faster than HTTP
- **Throughput**: 1000+ requests/second with batching
- **Latency**: Sub-50ms end-to-end response time
- **GPU Utilization**: 60-80% under load

## Production Deployment

### Infrastructure Requirements
- NVIDIA GPU (recommended for optimal performance)
- Docker containerization platform
- NVIDIA Container Toolkit for GPU access
- Minimum 2GB RAM, 1GB storage

### Deployment Architecture
- **Triton Server**: Containerized inference serving
- **Model Repository**: Organized model artifacts and configurations
- **Client Libraries**: Python SDK with gRPC/HTTP support
- **Monitoring**: Health checks and performance metrics

### API Endpoints
- **gRPC**: localhost:8001 (primary, high-performance)
- **HTTP**: localhost:8000 (fallback, web-compatible)
- **Metrics**: localhost:8002 (monitoring and diagnostics)

## Technical Implementation

### Model Training Pipeline
```bash
cd training/

# Install training dependencies
pip install -r requirements.training.txt

# Data collection and preprocessing (images + annotations.csv) -> generates keypoints and labels.npy which gets passed to the MLP model.
python data_collector.py

# Model training with cross-validation
python train_mlp.py

# ONNX export for production deployment
python export_onnx.py
```

### Production Deployment
```bash
# Single-command deployment
./deployment/deploy.sh

# Health verification
python clients/triton_client.py --health-check

# Basic inference (gRPC)
python clients/triton_client.py --image path/to/image.jpg

# Batch processing with benchmark
python clients/triton_client.py --image image.jpg --benchmark 20

# HTTP fallback  
python clients/triton_client.py --image image.jpg --use-http

# gRPC-only 
python clients/client_grpc.py --image image.jpg --benchmark 20
```

### API Usage Example
```python
# gRPC client request
response = triton_client.infer(
    model_name="ensemble_phone_detection",
    inputs=inputs,
    outputs=outputs
)

# Response format
{
  "person_id": 0,
  "is_phone": true,
  "confidence": 0.9525
}
```
### Future Enhancements
1. **TensorRT Integration**: Additional 2x performance improvement on NVIDIA hardware
2. **Multi-person Processing**: Extend beyond single-person detection
3. **Real-time Streaming**: WebSocket or gRPC streaming for video processing
4. **Edge Deployment**: Optimization for resource-constrained environments
