#!/usr/bin/env python3

import argparse
import numpy as np
import cv2
import requests
import json
import time
from pathlib import Path
import base64

# Try to import gRPC client, fall back to HTTP-only if not available
try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import *
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    print("gRPC client not available")

def load_image(image_path):
    """Load and encode image for Triton inference."""
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Encode image to bytes (JPEG format)
        _, encoded_image = cv2.imencode('.jpg', image)
        image_bytes = encoded_image.tobytes()
        
        return image_bytes, image.shape
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

def send_grpc_request(image_bytes, triton_url="localhost:8001", model_name="ensemble_phone_detection"):
    """Send gRPC inference request to Triton server."""
    
    if not GRPC_AVAILABLE:
        print("gRPC client not available")
        return None
    
    try:
        # Create gRPC client
        triton_client = grpcclient.InferenceServerClient(url=triton_url)
        
        # Prepare input data
        input_data = np.array([list(image_bytes)], dtype=np.uint8)
        
        # Create input object
        inputs = [grpcclient.InferInput("INPUT_IMAGE", input_data.shape, "UINT8")]
        inputs[0].set_data_from_numpy(input_data)
        
        # Define expected outputs
        outputs = [
            grpcclient.InferRequestedOutput("PERSON_ID"),
            grpcclient.InferRequestedOutput("IS_PHONE"),
            grpcclient.InferRequestedOutput("CONFIDENCE")
        ]
        
        # Send inference request
        response = triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        return response
        
    except Exception as e:
        print(f"Error sending gRPC request: {e}")
        return None

def send_http_request(image_bytes, triton_url="http://localhost:8000"):
    """Send HTTP inference request to Triton server."""
    
    # Prepare the request payload
    payload = {
        "inputs": [
            {
                "name": "INPUT_IMAGE",
                "shape": [1, len(image_bytes)],  # [batch_size, image_bytes]
                "datatype": "UINT8",
                "data": list(image_bytes)  # Convert bytes to list of integers
            }
        ],
        "outputs": [
            {"name": "PERSON_ID"},
            {"name": "IS_PHONE"}, 
            {"name": "CONFIDENCE"}
        ]
    }
    
    # Send HTTP request
    url = f"{triton_url}/v2/models/ensemble_phone_detection/infer"
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending HTTP request: {e}")
        return None

def send_triton_request(image_bytes, triton_url="localhost:8001", use_grpc=True):
    """Send inference request to Triton server using gRPC (default) or HTTP."""
    
    if use_grpc and GRPC_AVAILABLE:
        return send_grpc_request(image_bytes, triton_url)
    else:
        # Convert gRPC URL to HTTP if needed
        if not triton_url.startswith('http'):
            triton_url = f"http://{triton_url.replace(':8001', ':8000')}"
        return send_http_request(image_bytes, triton_url)

def parse_triton_response(response, is_grpc=True):
    """Parse Triton inference response (supports both gRPC and HTTP)."""
    if response is None:
        return []
    
    try:
        if is_grpc and hasattr(response, 'as_numpy'):
            # gRPC response parsing
            person_ids = response.as_numpy("PERSON_ID")
            is_phone = response.as_numpy("IS_PHONE") 
            confidence = response.as_numpy("CONFIDENCE")
            
            # Combine results
            results = []
            if person_ids is not None and is_phone is not None and confidence is not None:
                for i in range(len(person_ids)):
                    results.append({
                        'person_id': int(person_ids[i]),
                        'is_phone': bool(is_phone[i]),
                        'confidence': float(confidence[i])
                    })
            
            return results
            
        else:
            # HTTP response parsing
            if not response or 'outputs' not in response:
                return []
                
            outputs = response['outputs']
            
            # Extract outputs by name
            person_ids = None
            is_phone = None
            confidence = None
            
            for output in outputs:
                if output['name'] == 'PERSON_ID':
                    person_ids = output['data']
                elif output['name'] == 'IS_PHONE':
                    is_phone = output['data']
                elif output['name'] == 'CONFIDENCE':
                    confidence = output['data']
            
            # Combine results
            results = []
            if person_ids and is_phone and confidence:
                for i in range(len(person_ids)):
                    results.append({
                        'person_id': person_ids[i],
                        'is_phone': bool(is_phone[i]),
                        'confidence': float(confidence[i])
                    })
            
            return results
        
    except Exception as e:
        print(f"Error parsing response: {e}")
        return []

def test_single_image(image_path, triton_url="localhost:8001", verbose=False, use_grpc=True):
    """Test phone detection on a single image."""
    
    protocol = "gRPC" if use_grpc and GRPC_AVAILABLE else "HTTP"
    print(f"Testing image: {image_path} ({protocol})")
    
    # Load image
    image_bytes, image_shape = load_image(image_path)
    if image_bytes is None:
        return False
    
    if verbose:
        print(f"   Image shape: {image_shape}")
        print(f"   Image size: {len(image_bytes)} bytes")
    
    # Send inference request
    print(f"Sending {protocol} request to Triton server...")
    response = send_triton_request(image_bytes, triton_url, use_grpc)
    
    if response is None:
        print("Failed to get response from server")
        return False
    
    # Parse results
    is_grpc_response = use_grpc and GRPC_AVAILABLE and hasattr(response, 'as_numpy')
    results = parse_triton_response(response, is_grpc_response)
    
    if not results:
        print("No people detected or processing error")
        return True
    
    # Display results
    print(f"Detected {len(results)} person(s):")
    for result in results:
        status = "using a phone" if result['is_phone'] else "NOT using a phone"
        confidence = result['confidence']
        person_id = result['person_id']
        
        print(f"      Person {person_id}: {status} (Confidence: {confidence:.4f})")
    
    return True

def test_server_health(triton_url="localhost:8001", use_grpc=True):
    """Test if Triton server is running and models are loaded."""
    
    protocol = "gRPC" if use_grpc and GRPC_AVAILABLE else "HTTP"
    print(f"Testing server health ({protocol})...")
    
    try:
        if use_grpc and GRPC_AVAILABLE:
            # gRPC health check
            triton_client = grpcclient.InferenceServerClient(url=triton_url)
            
            if triton_client.is_server_ready():
                print("Server is ready")
            else:
                print("Server is not ready")
                return False
                
            if triton_client.is_server_live():
                print(" Server is live")
            else:
                print(" Server is not live")
                return False
        else:
            # HTTP health check
            if not triton_url.startswith('http'):
                triton_url = f"http://{triton_url.replace(':8001', ':8000')}"
                
            health_url = f"{triton_url}/v2/health/ready"
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                print("  Server is healthy")
            else:
                print(f" Server health check returned status: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"    Server health check failed: {e}")
        return False
    
    # Test model availability
    print("Checking model availability...")
    
    models_to_check = [
        "preprocess",
        "mlp_phone_detector", 
        "postprocess",
        "ensemble_phone_detection"
    ]
    
    for model_name in models_to_check:
        try:
            if use_grpc and GRPC_AVAILABLE:
                # gRPC model check
                if triton_client.is_model_ready(model_name):
                    print(f"{model_name}: Ready")
                else:
                    print(f"{model_name}: Not ready")
                    return False
            else:
                # HTTP model check
                if not triton_url.startswith('http'):
                    triton_url = f"http://{triton_url.replace(':8001', ':8000')}"
                    
                model_url = f"{triton_url}/v2/models/{model_name}"
                response = requests.get(model_url, timeout=5)
                
                if response.status_code == 200:
                    model_info = response.json()
                    if model_info.get('state') == 'READY':
                        print(f"{model_name}: Ready")
                    else:
                        print(f"{model_name}: {model_info.get('state', 'Unknown state')}")
                        return False
                else:
                    print(f"{model_name}: Not available (Status: {response.status_code})")
                    return False
                
        except Exception as e:
            print(f"{model_name}: Check failed - {e}")
            return False
    
    return True

def benchmark_performance(image_path, triton_url="localhost:8001", num_requests=10, use_grpc=True):
    """Benchmark inference performance."""
    
    protocol = "gRPC" if use_grpc and GRPC_AVAILABLE else "HTTP"
    print(f"\nBenchmarking {protocol} performance with {num_requests} requests...")
    
    # Load test image
    image_bytes, _ = load_image(image_path)
    if image_bytes is None:
        return
    
    times = []
    successful_requests = 0
    
    # Warm up request
    print("Warming up...")
    send_triton_request(image_bytes, triton_url, use_grpc)
    
    print(f"Running {num_requests} benchmark requests...")
    
    for i in range(num_requests):
        start_time = time.time()
        
        response = send_triton_request(image_bytes, triton_url, use_grpc)
        
        end_time = time.time()
        
        if response is not None:
            successful_requests += 1
            times.append(end_time - start_time)
            
        if (i + 1) % 5 == 0:
            print(f"     Completed {i + 1}/{num_requests} requests...")
    
    if times:
        avg_time = np.mean(times) * 1000  # Convert to ms
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        std_time = np.std(times) * 1000
        p95_time = np.percentile(times, 95) * 1000
        p99_time = np.percentile(times, 99) * 1000
        
        print(f"\n{protocol} Performance Results:")
        print(f"   Successful requests: {successful_requests}/{num_requests}")
        print(f"   Average latency: {avg_time:.2f} ms")
        print(f"   Min latency: {min_time:.2f} ms")
        print(f"   Max latency: {max_time:.2f} ms")
        print(f"   95th percentile: {p95_time:.2f} ms") 
        print(f"   99th percentile: {p99_time:.2f} ms")
        print(f"   Std deviation: {std_time:.2f} ms")
        print(f"   Throughput: {1000 / avg_time:.2f} requests/second")
        
        return {
            'protocol': protocol,
            'avg_time_ms': avg_time,
            'throughput_rps': 1000 / avg_time,
            'successful_requests': successful_requests,
            'total_requests': num_requests
        }
    else:
        print("    No successful requests for benchmarking")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Test Triton Phone Detection Pipeline"
    )
    parser.add_argument(
        "--image", 
        type=str,
        help="Path to test image file"
    )
    parser.add_argument(
        "--triton-url",
        type=str, 
        default="localhost:8001",
        help="Triton server URL (default: localhost:8001 for gRPC)"
    )
    parser.add_argument(
        "--use-http",
        action="store_true",
        help="Use HTTP instead of gRPC (default: gRPC)"
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Only run server health check"
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        help="Run performance benchmark with N requests"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Determine protocol
    use_grpc = not args.use_http and GRPC_AVAILABLE
    protocol = "gRPC" if use_grpc else "HTTP"
    
    print("=" * 60)
    print(f" TRITON PHONE DETECTION PIPELINE TESTER ({protocol})")
    print("=" * 60)
    
    if not GRPC_AVAILABLE and not args.use_http:
        print(" gRPC not available, falling back to HTTP")
        use_grpc = False
    
    # Always run health check first
    if not test_server_health(args.triton_url, use_grpc):
        print("\nServer health check failed. Please start Triton server and ensure models are loaded.")
        return 1
    
    if args.health_check:
        print("\n Health check completed successfully!")
        return 0
    
    # Test with image if provided
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"\nImage file not found: {image_path}")
            return 1
        
        success = test_single_image(image_path, args.triton_url, args.verbose, use_grpc)
        if not success:
            return 1
        
        # Run benchmark if requested
        if args.benchmark:
            benchmark_performance(image_path, args.triton_url, args.benchmark, use_grpc)
    
    else:
        print("\n Usage examples:")
        print(f"   python {__file__} --health-check")
        print(f"   python {__file__} --image path/to/test/image.jpg")
        print(f"   python {__file__} --image path/to/test/image.jpg --benchmark 10")
    
    print("\n Test completed!")
    return 0

if __name__ == "__main__":
    exit(main())