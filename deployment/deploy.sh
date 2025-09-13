#!/bin/bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Deploying Phone Detection Model to Triton Inference Server"

# Configuration
MODEL_REPO_DIR="$SCRIPT_DIR/../models/model_repository"
CUSTOM_TRITON_IMAGE="phone-detection-server:latest"
CONTAINER_NAME="phone-detection"
HTTP_PORT=8000
GRPC_PORT=8001
METRICS_PORT=8002

# Check if Docker is running
check_docker() {
    echo -e "Checking Docker..."
    if ! docker info > /dev/null 2>&1;
    then
        echo -e "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    echo -e "Docker is running"
}

# Check if model repository exists
check_model_repository() {
    echo -e "Checking model repo"
    if [ ! -d "$MODEL_REPO_DIR" ]; then
        echo -e "Model repo not found at $MODEL_REPO_DIR"
        exit 1
    fi

    required_models=("preprocess_img" "mlp_phone_detector" "postprocess" "ensemble_phone_detection" "rtmdet_detection" "person_cropper" "rtmpose_estimation" "feature_normalizer")
    for model in "${required_models[@]}"; do
        if [ ! -d "$MODEL_REPO_DIR/$model" ]; then
            echo -e "Missing model: $model"
            exit 1
        fi
    done

    echo -e "Model repository is ready"
}

# New function to build our custom image from the Dockerfile
build_image() {
    echo "Building custom Triton image with Python dependencies..."
    if [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
        echo "Using docker-compose to build image..."
        docker compose -f "$SCRIPT_DIR/docker-compose.yml" build
        echo "Custom image built successfully via docker-compose"
    else
        echo "Using docker build..."
        docker build -t $CUSTOM_TRITON_IMAGE -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"
        echo "Custom image built successfully"
    fi
}

# Stop and remove existing container

cleanup_existing() {
    echo -e "Cleaning up existing containers"

    # Stop container if it is running, but don't exit if it fails
    echo "Attempting to stop existing container..."
    docker stop $CONTAINER_NAME || true

    # Remove container if it exists, but don't exit if it fails
    echo "Attempting to remove existing container..."
    docker rm $CONTAINER_NAME || true

    echo -e "Cleanup completed"
}
# Start Triton server
start_triton() {
    echo -e "Starting triton server"

    if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
        echo -e "GPU detected, enabling GPU acceleration"
        GPU_ARGS="--gpus all"
    else
        echo -e "No GPU detected, running on CPU only"
        GPU_ARGS=""
    fi

    # Create logs directory
    mkdir -p $SCRIPT_DIR/logs

    # Choose deployment method based on available files
    if [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
        echo "Using docker-compose to start container..."
        docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d
        echo -e "Triton server container started via docker-compose"
    else
        echo "Using docker run to start container..."
        docker run --privileged --gpus all -d --rm --it \
            --name $CONTAINER_NAME \
            -p $HTTP_PORT:8000 \
            -p $GRPC_PORT:8001 \
            -p $METRICS_PORT:8002 \
            -v "$MODEL_REPO_DIR:/models:ro" \
            -v "$SCRIPT_DIR/logs:/logs" \
            $CUSTOM_TRITON_IMAGE \
            tritonserver \
                --model-repository=/models \
                --allow-http=true \
                --allow-grpc=true \
                --allow-metrics=true \
                --http-port=8000 \
                --grpc-port=8001 \
                --metrics-port=8002 \
                --log-verbose=1 \
                --exit-on-error=false \
                --strict-model-config=false
        echo -e "Triton server container started via docker run"
    fi
}

# Wait for server to be ready
wait_for_server() {
    echo -e "Waiting for triton server to be ready"

    MAX_ATTEMPTS=90  # Increased timeout for MMPose model loading
    ATTEMPT=1

    while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
        if curl -s -f "http://localhost:$HTTP_PORT/v2/health/ready" > /dev/null 2>&1;
        then
            echo -e "\nTriton server is ready"
            return 0
        fi

        echo -n "."
        sleep 3  # Increased sleep time
        ATTEMPT=$((ATTEMPT + 1))
    done

    echo -e "\nServer failed to become ready"
    echo -e "Checking container logs for debugging..."
    docker logs --tail 20 $CONTAINER_NAME
    return 1
}

# Test server
test_server() {
    echo "Testing server endpoints..."

    # Test health endpoint
    if curl -s -f "http://localhost:$HTTP_PORT/v2/health/ready" > /dev/null;
    then
        echo -e "Health endpoint is working"
    else
        echo -e "Health endpoint failed"
        return 1
    fi

    # Test model availability with retries for slow loading models
    models=("mlp_phone_detector" "postprocess" "preprocess" "ensemble_phone_detection")
    for model in "${models[@]}"; do
        echo "Checking model: $model"
        RETRIES=10
        for i in $(seq 1 $RETRIES);
        do
            if curl -s -f "http://localhost:$HTTP_PORT/v2/models/$model" > /dev/null;
            then
                echo -e "Model '$model' is available"
                break
            elif [ $i -eq $RETRIES ]; then
                echo -e "Model '$model' is not available after $RETRIES attempts"
                echo -e "Checking model status..."
                curl -s "http://localhost:$HTTP_PORT/v2/models/$model" 2>/dev/null || echo "Model not found"
                return 1
            else
                echo -n "."
                sleep 5
            fi
        done
    done

    return 0
}

# Check that all models are loaded and ready
verify_models_loaded() {
    echo "Verifying all models are loaded and ready..."

    # Wait for logs to show all models loaded
    echo "Waiting for all models to finish loading..."
    TIMEOUT=120
    START_TIME=$(date +%s)

    while true; do
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))

        if [ $ELAPSED -gt $TIMEOUT ]; then
            echo -e "Timeout waiting for models to load"
            return 1
        fi

        # Check if all required models are loaded
        LOADED_COUNT=$(docker logs $CONTAINER_NAME 2>&1 | grep "successfully loaded" | wc -l)

        if [ $LOADED_COUNT -ge 3 ]; then
            echo -e "All 3 models loaded successfully"
            break
        else
            echo "Models loaded: $LOADED_COUNT/3"
            sleep 5
        fi
    done

    return 0
}

show_info() {
    echo -e "-----------------------------------------"
    echo -e "HTTP Endpoint:    http://localhost:$HTTP_PORT"
    echo -e "gRPC Endpoint:    localhost:$GRPC_PORT"
    echo -e "Metrics Endpoint: http://localhost:$METRICS_PORT/metrics"
    echo -e "Health Check:     http://localhost:$HTTP_PORT/v2/health/ready"
    echo -e "-----------------------------------------"
    echo -e "Models loaded:"
    echo -e "  • preprocess (MMPose + Python)"
    echo -e "  • mlp_phone_detector (ONNX + GPU)"
    echo -e "  • postprocess (Python)"
    echo -e "  • ensemble_phone_detection (Pipeline)"
    echo -e "-----------------------------------------"
    echo -e "Commands:"
    echo -e "  View logs:        ./deploy.sh logs"
    echo -e "  Stop server:      ./deploy.sh stop"
    echo -e "  Check status:     ./deploy.sh status"
    echo -e "-----------------------------------------"
}

# Main deployment function
main() {
    case "${1:-deploy}" in
        "deploy"|"start")
            check_docker
            check_model_repository
            build_image
            cleanup_existing
            start_triton
            verify_models_loaded && wait_for_server && test_server && show_info
            ;;
        "stop")
            echo -e "Stopping triton server"
            if [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
                docker compose -f "$SCRIPT_DIR/docker-compose.yml" down
            else
                docker stop $CONTAINER_NAME || true
            fi
            echo -e "Server stopped"
            ;;
        "restart")
            echo -e "Restarting triton server"
            if [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
                docker compose -f "$SCRIPT_DIR/docker-compose.yml" restart
            else
                docker restart $CONTAINER_NAME || true
            fi
            verify_models_loaded && wait_for_server
            echo -e "Server restarted"
            ;;
        "logs")
            if [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
                docker compose -f "$SCRIPT_DIR/docker-compose.yml" logs -f
            else
                docker logs -f $CONTAINER_NAME
            fi
            ;;
        "status")
            if docker ps -q -f name=$CONTAINER_NAME | grep -q . 2>/dev/null || docker ps -q -f name=$CONTAINER_NAME | grep -q . 2>/dev/null;
            then
                echo -e "Server is running"
                test_server
            else
                echo -e "Server is not running"
            fi
            ;;
        "clean")
            echo -e "Removing server container"
            if [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
                docker compose -f "$SCRIPT_DIR/docker-compose.yml" down --rmi all
            else
                docker rm -f $CONTAINER_NAME || true
            fi
            echo -e "Container removed"
            ;;
        "help"|"-h"|"--help")
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  deploy    Build, deploy and start the server (default)"
            echo "  start     Same as deploy"
            echo "  stop      Stop the server"
            echo "  restart   Restart the server"
            echo "  status    Check server status"
            echo "  logs      Follow server logs"
            echo "  clean     Remove server container"
            echo "  help      Show this help"
            ;;
        *)
            echo "Use '$0 help' for usage information."
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
