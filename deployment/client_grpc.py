#!/usr/bin/env python3

import argparse
import numpy as np
import cv2
import time
from pathlib import Path
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

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

def parse_grpc_response(response):
    """Parse Triton gRPC inference response."""
    if response is None:
        return []

    try:
        # Extract output data
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

    except Exception as e:
        print(f"Error parsing gRPC response: {e}")
        return []

def test_single_image(image_path, triton_url="localhost:8001", verbose=False):
    """Test phone detection on a single image using gRPC."""

    print(f"Testing image: {image_path}")

    # Load image
    image_bytes, image_shape = load_image(image_path)
    if image_bytes is None:
        return False

    if verbose:
        print(f"   Image shape: {image_shape}")
        print(f"   Image size: {len(image_bytes)} bytes")

    print("Sending gRPC request to Triton server...")
    response = send_grpc_request(image_bytes, triton_url)

    if response is None:
        print("Failed to get response from server")
        return False

    results = parse_grpc_response(response)

    if not results:
        print(" No people detected or processing error")
        return True

    # Display results
    print(f" Detected {len(results)} person(s):")
    for result in results:
        status = "using a phone" if result['is_phone'] else "NOT using a phone"
        confidence = result['confidence']
        person_id = result['person_id']

        print(f"Person {person_id}: {status} (Confidence: {confidence:.4f})")

    return True

def test_server_health(triton_url="localhost:8001"):
    """Test if Triton server is running and models are loaded via gRPC."""

    try:
        # Create gRPC client
        triton_client = grpcclient.InferenceServerClient(url=triton_url)

        # Test server readiness
        if triton_client.is_server_ready():
            print("Server is ready")
        else:
            print("Server is not ready")
            return False

        # Test server liveness
        if triton_client.is_server_live():
            print("Server is live")
        else:
            print("Server is not live")
            return False

    except Exception as e:
        print(f"Server health check failed: {e}")
        return False

    # Test model availability

    models_to_check = [
        "preprocess",
        "mlp_phone_detector",
        "postprocess",
        "ensemble_phone_detection"
    ]

    for model_name in models_to_check:
        try:
            if triton_client.is_model_ready(model_name):
                print(f"{model_name}: Ready")
            else:
                print(f"{model_name}: Not ready")
                return False

        except Exception as e:
            print(f"{model_name}: Check failed - {e}")
            return False

    return True

def benchmark_performance(image_path, triton_url="localhost:8001", num_requests=10):
    """Benchmark gRPC inference performance."""

    print(f"Benchmarking gRPC performance with {num_requests} requests...")

    # Load test image
    image_bytes, _ = load_image(image_path)
    if image_bytes is None:
        return

    times = []
    successful_requests = 0

    # Warm up request
    send_grpc_request(image_bytes, triton_url)

    print(f"Running {num_requests} benchmark requests...")

    for i in range(num_requests):
        start_time = time.time()

        response = send_grpc_request(image_bytes, triton_url)

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

        print("gRPC Performance Results:")
        print(f"   Successful requests: {successful_requests}/{num_requests}")
        print(f"   Average latency: {avg_time:.2f} ms")
        print(f"   Min latency: {min_time:.2f} ms")
        print(f"   Max latency: {max_time:.2f} ms")
        print(f"   95th percentile: {p95_time:.2f} ms")
        print(f"   99th percentile: {p99_time:.2f} ms")
        print(f"   Std deviation: {std_time:.2f} ms")
        print(f"   Throughput: {1000 / avg_time:.2f} requests/second")

        return {
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'p95_time_ms': p95_time,
            'p99_time_ms': p99_time,
            'throughput_rps': 1000 / avg_time,
            'successful_requests': successful_requests,
            'total_requests': num_requests
        }
    else:
        print("No successful requests for benchmarking")
        return None

def main():
    """Main function for gRPC client testing."""

    parser = argparse.ArgumentParser(
        description="Test Triton Phone Detection Pipeline via gRPC"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to test image file"
    )
    parser.add_argument(
        "--triton-host",
        type=str,
        default="localhost",
        help="Triton server host (default: localhost)"
    )
    parser.add_argument(
        "--grpc-url",
        type=str,
        default="{}:8001",
        help="Triton gRPC server URL (default: {}:8001)"
    )
    parser.add_argument(
        "--http-url",
        type=str,
        default="http://localhost:8000",
        help="Triton HTTP server URL for comparison (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Only run server health check"
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=10,
        help="Run performance benchmark with N requests (default: 10)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    grpc_url = args.grpc_url.format(args.triton_host)

    print("TRITON PHONE DETECTION - gRPC CLIENT")

    if not test_server_health(grpc_url):
        print("\nServer health check failed. Please start Triton server and ensure models are loaded.")
        return 1

    if args.health_check:
        print("\nHealth check completed successfully!")
        return 0

    # Test with image if provided
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"\nImage file not found: {image_path}")
            return 1

        # Single image test
        success = test_single_image(image_path, grpc_url, args.verbose)
        if not success:
            return 1

        # Run benchmark
        if args.benchmark:
            benchmark_performance(image_path, grpc_url, args.benchmark)

    else:
        print("Usage examples:")
        print(f"   python {__file__} --health-check")
        print(f"   python {__file__} --image path/to/test/image.jpg")
        print(f"   python {__file__} --image image.jpg --benchmark 20")
        print(f"   python {__file__} --image image.jpg --compare")

    print("GRPC TEST COMPLETED")
    return 0


if __name__ == "__main__":
    exit(main())
