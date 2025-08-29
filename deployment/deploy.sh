#!/bin/bash

# Phone Detection Triton Deployment Script
set -e

echo "Deploying Phone Detection Model to Triton Inference Server"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration  
MODEL_REPO_DIR="../models/model_repository"
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:22.12-py3"
CONTAINER_NAME="phone-detection-triton"
HTTP_PORT=8000
GRPC_PORT=8001
METRICS_PORT=8002

# Check if Docker is running
check_docker() {
    echo -e "${BLUE}Checking Docker...${NC}"
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Docker is not running. Please start Docker and try again.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Docker is running${NC}"
}

# Check if model repository exists
check_model_repository() {
    echo -e "${BLUE}Checking model repository...${NC}"
    if [ ! -d "$MODEL_REPO_DIR" ]; then
        echo -e "${RED}Model repository not found at $MODEL_REPO_DIR${NC}"
        echo -e "${YELLOW}Please run the model export and setup scripts first.${NC}"
        exit 1
    fi
    
    # Check for required models
    required_models=("preprocess" "mlp_phone_detector" "postprocess" "ensemble_phone_detection")
    for model in "${required_models[@]}"; do
        if [ ! -d "$MODEL_REPO_DIR/$model" ]; then
            echo -e "${RED} Missing model: $model${NC}"
            exit 1
        fi
    done
    
    echo -e "${GREEN}✅ Model repository is ready${NC}"
}

# Pull Triton image
pull_triton_image() {
    echo -e "${BLUE}Pulling Triton Inference Server image...${NC}"
    docker pull $TRITON_IMAGE
    echo -e "${GREEN}Triton image pulled${NC}"
}

# Stop and remove existing container
cleanup_existing() {
    echo -e "${BLUE}Cleaning up existing containers...${NC}"
    
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        echo "Stopping existing container..."
        docker stop $CONTAINER_NAME
    fi
    
    if docker ps -a -q -f name=$CONTAINER_NAME | grep -q .; then
        echo "Removing existing container..."
        docker rm $CONTAINER_NAME
    fi
    
    echo -e "${GREEN}Cleanup completed${NC}"
}

# Start Triton server
start_triton() {
    echo -e "${BLUE}Starting Triton Inference Server...${NC}"
    
    # Check if GPU is available
    GPU_ARGS=""
    if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
        echo -e "${GREEN}GPU detected, enabling GPU acceleration${NC}"
        GPU_ARGS="--gpus all"
    else
        echo -e "${YELLOW}  No GPU detected, running on CPU only${NC}"
    fi
    
    # Create logs directory
    mkdir -p ./logs
    
    # Start container
    docker run -d \
        --name $CONTAINER_NAME \
        $GPU_ARGS \
        -p $HTTP_PORT:8000 \
        -p $GRPC_PORT:8001 \
        -p $METRICS_PORT:8002 \
        -v "$(pwd)/$MODEL_REPO_DIR:/models:ro" \
        -v "$(pwd)/logs:/logs" \
        $TRITON_IMAGE \
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
    
    echo -e "${GREEN}✅ Triton server container started${NC}"
}

# Wait for server to be ready
wait_for_server() {
    echo -e "${BLUE}Waiting for Triton server to be ready...${NC}"
    
    MAX_ATTEMPTS=60
    ATTEMPT=1
    
    while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
        if curl -s -f "http://localhost:$HTTP_PORT/v2/health/ready" > /dev/null 2>&1; then
            echo -e "${GREEN}Triton server is ready!${NC}"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ATTEMPT=$((ATTEMPT + 1))
    done
    
    echo -e "\n${RED}Server failed to become ready after $((MAX_ATTEMPTS * 2)) seconds${NC}"
    echo -e "${YELLOW}Check server logs: docker logs $CONTAINER_NAME${NC}"
    return 1
}

# Test server
test_server() {
    echo -e "${BLUE}Testing server endpoints...${NC}"
    
    # Test health endpoint
    if curl -s -f "http://localhost:$HTTP_PORT/v2/health/ready" > /dev/null; then
        echo -e "${GREEN}Health endpoint working${NC}"
    else
        echo -e "${RED}Health endpoint failed${NC}"
        return 1
    fi
    
    # Test model availability
    models=("preprocess" "mlp_phone_detector" "postprocess" "ensemble_phone_detection")
    for model in "${models[@]}"; do
        if curl -s -f "http://localhost:$HTTP_PORT/v2/models/$model" > /dev/null; then
            echo -e "${GREEN} Model $model is available${NC}"
        else
            echo -e "${RED} Model $model is not available${NC}"
            return 1
        fi
    done
    
    return 0
}

# Show deployment info
show_info() {
    echo -e "\n${GREEN} Deployment completed successfully!${NC}"
    echo -e "\n${BLUE} Server Information:${NC}"
    echo -e "HTTP Endpoint:    http://localhost:$HTTP_PORT"
    echo -e "gRPC Endpoint:    localhost:$GRPC_PORT"
    echo -e "Metrics Endpoint: http://localhost:$METRICS_PORT/metrics"
    echo -e "Health Check:     http://localhost:$HTTP_PORT/v2/health/ready"
    
    echo -e "\n${BLUE} Testing Commands:${NC}"
    echo -e "Health check:     python ../clients/triton_client.py --health-check"
    echo -e "Test with image:  python ../clients/triton_client.py --image /path/to/image.jpg"
    echo -e "Performance test: python ../clients/triton_client.py --image /path/to/image.jpg --benchmark 10"
    
    echo -e "\n${BLUE} Management Commands:${NC}"
    echo -e "View logs:        docker logs -f $CONTAINER_NAME"
    echo -e "Stop server:      docker stop $CONTAINER_NAME"
    echo -e "Start server:     docker start $CONTAINER_NAME"
    echo -e "Remove server:    docker rm -f $CONTAINER_NAME"
    
    echo -e "\n${BLUE} Model Repository:${NC} $MODEL_REPO_DIR"
    echo -e "${BLUE} Server Logs:${NC} ./logs/"
}

# Main deployment function
main() {
    case "${1:-deploy}" in
        "deploy"|"start")
            check_docker
            check_model_repository
            pull_triton_image
            cleanup_existing
            start_triton
            wait_for_server
            test_server
            show_info
            ;;
        "stop")
            echo -e "${BLUE}Stopping Triton server...${NC}"
            docker stop $CONTAINER_NAME || true
            echo -e "${GREEN}Server stopped${NC}"
            ;;
        "restart")
            echo -e "${BLUE}Restarting Triton server...${NC}"
            docker restart $CONTAINER_NAME || true
            wait_for_server
            echo -e "${GREEN} Server restarted${NC}"
            ;;
        "logs")
            docker logs -f $CONTAINER_NAME
            ;;
        "status")
            if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
                echo -e "${GREEN} Server is running${NC}"
                test_server
            else
                echo -e "${RED} Server is not running${NC}"
            fi
            ;;
        "clean")
            echo -e "${BLUE}Removing server container...${NC}"
            docker rm -f $CONTAINER_NAME || true
            echo -e "${GREEN} Container removed${NC}"
            ;;
        "help"|"-h"|"--help")
            echo -e "${BLUE}Phone Detection Triton Deployment Script${NC}"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  deploy    Deploy and start the server (default)"
            echo "  start     Same as deploy"
            echo "  stop      Stop the server"
            echo "  restart   Restart the server"
            echo "  status    Check server status"
            echo "  logs      Show server logs"
            echo "  clean     Remove server container"
            echo "  help      Show this help"
            ;;
        *)
            echo -e "${RED}Unknown command: $1${NC}"
            echo "Use '$0 help' for usage information."
            exit 1
            ;;
    esac
}

# Run main function
main "$@"