#!/usr/bin/env python3
"""
Usage:
    from predict import PhoneDetectionClient
    
    detector = PhoneDetectionClient("localhost:8001")
    result = detector.detect_phone_usage("path/to/image.jpg")
    print(result)  # {"person_id": 0, "is_phone": true, "confidence": 0.8547}
"""

import tritonclient.grpc as grpcclient
import numpy as np
import cv2
import logging
from typing import Dict, Union, Optional
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhoneDetectionClient:
    def __init__(self, triton_url: str = "localhost:8001", model_name: str = "ensemble_phone_detection"):
        """
        Initialize connection to Triton server.
        
        Args:
            triton_url: URL of Triton server (host:port format)
            model_name: Name of the ensemble model to use
        """
        self.triton_url = triton_url
        self.model_name = model_name
        self.client = None
        self._connect()
        
    def _connect(self):
        """Establish connection to Triton server."""
        try:
            self.client = grpcclient.InferenceServerClient(url=self.triton_url)
            # Test connection
            if self.client.is_server_ready():
                logger.info(f"Successfully connected to Triton server at {self.triton_url}")
            else:
                raise ConnectionError("Server is not ready")
        except Exception as e:
            logger.error(f"Failed to connect to Triton server at {self.triton_url}: {e}")
            raise
    
    def health_check(self) -> Dict:
        """ Returns dict with server and model status information """
        try:
            server_ready = self.client.is_server_ready()
            model_ready = self.client.is_model_ready(self.model_name)
            
            return {
                "server_ready": server_ready,
                "model_ready": model_ready,
                "triton_url": self.triton_url,
                "model_name": self.model_name,
                "status": "healthy" if (server_ready and model_ready) else "unhealthy"
            }
        except Exception as e:
            return {
                "server_ready": False,
                "model_ready": False,
                "triton_url": self.triton_url,
                "model_name": self.model_name,
                "status": "error",
                "error": str(e)
            }
    
    def detect_phone_usage(self, image_input: Union[str, Path, np.ndarray], timeout_seconds: float = 30.0) -> Dict:
        """
        Detect phone usage in an image.
        
        Args:
            image_input: Path to image file or numpy array (BGR format)
            timeout_seconds: Request timeout in seconds
            
        Returns:
            Dict with detection results:
            {
                "person_id": int,
                "is_phone": bool, 
                "confidence": float,
                "inference_time_ms": float,
                "status": "success" | "error"
            }
        """
        start_time = time.time()
        
        try:
            # Load and prepare image
            if isinstance(image_input, (str, Path)):
                image = cv2.imread(str(image_input))
                if image is None:
                    raise ValueError(f"Could not load image: {image_input}")
            elif isinstance(image_input, np.ndarray):
                image = image_input
            else:
                raise ValueError("image_input must be file path or numpy array")
                
            # Encode image to bytes
            _, encoded = cv2.imencode('.jpg', image)
            image_bytes = encoded.tobytes()
            
            # Prepare Triton input (match the format used in client_grpc.py)
            input_data = np.array([list(image_bytes)], dtype=np.uint8)
            inputs = [grpcclient.InferInput("INPUT_IMAGE", input_data.shape, "UINT8")]
            inputs[0].set_data_from_numpy(input_data)
            
            # Define expected outputs
            outputs = [
                grpcclient.InferRequestedOutput("PERSON_ID"),
                grpcclient.InferRequestedOutput("IS_PHONE"),
                grpcclient.InferRequestedOutput("CONFIDENCE")
            ]
            
            # Send inference request  
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            # Extract results (matching client_grpc.py format)
            person_id = response.as_numpy("PERSON_ID")[0]
            is_phone = response.as_numpy("IS_PHONE")[0]
            confidence = response.as_numpy("CONFIDENCE")[0]
            
            inference_time = (time.time() - start_time) * 1000  # convert to milliseconds
            
            result = {
                "person_id": int(person_id),
                "is_phone": bool(is_phone),
                "confidence": float(confidence),
                "inference_time_ms": round(inference_time, 2),
                "status": "success"
            }
            
            logger.info(f"Detection completed: {result}")
            return result
            
        except Exception as e:
            error_result = {
                "person_id": -1,
                "is_phone": False,
                "confidence": 0.0,
                "inference_time_ms": round((time.time() - start_time) * 1000, 2),
                "status": "error",
                "error": str(e)
            }
            logger.error(f"Detection failed: {error_result}")
            return error_result
    
    def detect_batch(self, image_paths: list, timeout_seconds: float = 60.0) -> list:
        """
        Process multiple images in sequence.
        
        Args:
            image_paths: List of image file paths
            timeout_seconds: Total timeout for all requests
            
        Returns:
            List of detection results for each image
        """
        results = []
        start_time = time.time()
        
        for i, image_path in enumerate(image_paths):
            if time.time() - start_time > timeout_seconds:
                remaining_results = [{
                    "person_id": -1,
                    "is_phone": False,
                    "confidence": 0.0,
                    "inference_time_ms": 0.0,
                    "status": "timeout",
                    "error": "Batch timeout exceeded"
                }] * (len(image_paths) - i)
                results.extend(remaining_results)
                break
                
            result = self.detect_phone_usage(image_path, timeout_seconds=10.0)
            results.append(result)
            
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get model metadata and configuration.
        
        Returns:
            Dict with model information
        """
        try:
            metadata = self.client.get_model_metadata(self.model_name)
            config = self.client.get_model_config(self.model_name)
            
            return {
                "model_name": metadata.name,
                "platform": metadata.platform,
                "versions": metadata.versions,
                "inputs": [{"name": inp.name, "datatype": inp.datatype, "shape": inp.shape} 
                          for inp in metadata.inputs],
                "outputs": [{"name": out.name, "datatype": out.datatype, "shape": out.shape}
                           for out in metadata.outputs],
                "max_batch_size": config.max_batch_size,
                "status": "available"
            }
        except Exception as e:
            return {
                "model_name": self.model_name,
                "status": "error",
                "error": str(e)
            }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup if needed."""
        pass


def main():
    """Example usage and testing."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Production Phone Detection Client")
    parser.add_argument("--url", default="localhost:8001", help="Triton server URL")
    parser.add_argument("--image", help="Path to image for detection")
    parser.add_argument("--health", action="store_true", help="Run health check")
    parser.add_argument("--info", action="store_true", help="Get model info")
    parser.add_argument("--batch", nargs="+", help="Process multiple images")
    
    args = parser.parse_args()
    
    try:
        detector = PhoneDetectionClient(args.url)
    except Exception as e:
        print(f"Failed to initialize client: {e}")
        sys.exit(1)
    
    if args.health:
        health = detector.health_check()
        print(f"Health Check: {health}")
        return
    
    if args.info:
        info = detector.get_model_info()
        print(f"Model Info: {info}")
        return
    
    if args.image:
        result = detector.detect_phone_usage(args.image)
        print(f"Detection Result: {result}")
        return
    
    # Batch processing
    if args.batch:
        results = detector.detect_batch(args.batch)
        for i, result in enumerate(results):
            print(f"Image {i+1}: {result}")
        return

if __name__ == "__main__":
    main()