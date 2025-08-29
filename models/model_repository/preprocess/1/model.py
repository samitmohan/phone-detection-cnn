import json
import io
import numpy as np
import cv2
import triton_python_backend_utils as pb_utils
from PIL import Image

from mmpose.apis import MMPoseInferencer
MMPOSE_AVAILABLE = True  

class TritonPythonModel:
    """
    Preprocessing model for phone detection pipeline.
    
    This model:
    1. Receives raw image data
    2. Extracts human pose keypoints using MMPose
    3. Normalizes keypoints to create 51-element feature vectors
    4. Returns normalized features for MLP inference
    """

    def initialize(self, args):
        """Initialize the preprocessing model."""
        
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Get input/output configuration
        input_config = pb_utils.get_input_config_by_name(
            model_config, "INPUT_IMAGE")
        self.input_image_dtype = pb_utils.triton_string_to_numpy(
            input_config['data_type'])
        
        output_config = pb_utils.get_output_config_by_name(
            model_config, "NORMALIZED_FEATURES")
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type'])

        # Initialize MMPose inferencer if available
        if MMPOSE_AVAILABLE:
            try:
                self.pose_inferencer = MMPoseInferencer('human', device='cpu')
                print("MMPose inferencer initialized successfully")
            except Exception as e:
                print(f"Failed to initialize MMPose: {e}")
                self.pose_inferencer = None
                MMPOSE_AVAILABLE = False
        else:
            self.pose_inferencer = None

        # Keypoint indices for COCO-17 format
        self.NOSE, self.L_EYE, self.R_EYE, self.L_EAR, self.R_EAR = 0, 1, 2, 3, 4
        self.L_SHOULDER, self.R_SHOULDER = 5, 6
        self.L_HIP, self.R_HIP = 11, 12
        self.INPUT_DIM = 51  # 17 keypoints * 3 (x, y, confidence)

    def _get_kp_coords(self, kps, idx):
        """Returns keypoint coordinates if confidence is above threshold."""
        kp = kps[idx]
        return kp[:2] if kp[2] > 0.3 else None

    def _dist(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(p1 - p2)

    def _preprocess_keypoints(self, kps_raw):
        """
        Normalize keypoints to create pose-invariant features.
        
        Args:
            kps_raw: Raw keypoints array of shape [17, 3] (x, y, confidence)
            
        Returns:
            Normalized 51-element feature vector or None if invalid pose
        """
        l_shoulder = self._get_kp_coords(kps_raw, self.L_SHOULDER)
        r_shoulder = self._get_kp_coords(kps_raw, self.R_SHOULDER)
        l_hip = self._get_kp_coords(kps_raw, self.L_HIP)
        r_hip = self._get_kp_coords(kps_raw, self.R_HIP)

        # Calculate neck proxy (midpoint between shoulders)
        neck_proxy = None
        if l_shoulder is not None and r_shoulder is not None:
            neck_proxy = ((l_shoulder[0] + r_shoulder[0]) / 2, 
                         (l_shoulder[1] + r_shoulder[1]) / 2)
        
        # Calculate body scale (shoulder or hip width)
        body_scale = 0
        if l_shoulder is not None and r_shoulder is not None:
            body_scale = self._dist(l_shoulder, r_shoulder)
        elif l_hip is not None and r_hip is not None:
            body_scale = self._dist(l_hip, r_hip) 
        
        # Skip if we can't establish proper normalization
        if neck_proxy is None or body_scale < 1e-6: 
            return None 

        # Normalize all keypoints
        normalized_kps = []
        for i in range(kps_raw.shape[0]): 
            kp_coords = self._get_kp_coords(kps_raw, i)
            kp_score = kps_raw[i, 2]

            if kp_coords is not None:
                # Center relative to neck and scale by body size
                centered_kp = np.array(kp_coords) - np.array(neck_proxy)
                scaled_kp = centered_kp / body_scale
                normalized_kps.extend(scaled_kp.tolist()) 
            else:
                # Use zeros for missing keypoints
                normalized_kps.extend([0.0, 0.0]) 
            
            # Add confidence score
            normalized_kps.append(kp_score) 
        
        # Verify correct dimension
        if len(normalized_kps) == self.INPUT_DIM:
            return np.array(normalized_kps, dtype=np.float32)
        else:
            return None 

    def _extract_keypoints_mmpose(self, image_data):
        """Extract keypoints using MMPose."""
        if not MMPOSE_AVAILABLE or self.pose_inferencer is None:
            return self._create_mock_keypoints()

        try:
            # Convert image data to temporary file for MMPose
            # In production, you might want to optimize this
            temp_path = "/tmp/temp_image.jpg"
            cv2.imwrite(temp_path, image_data)
            
            # Run pose estimation
            result_generator = self.pose_inferencer(temp_path, return_vis=False)
            result = next(result_generator)
            predictions = result['predictions'][0]
            
            if not predictions:
                return []
            
            # Extract normalized features for each person
            features_list = []
            for pred in predictions:
                kps_raw = np.array([
                    (kp[0], kp[1], score) 
                    for kp, score in zip(pred['keypoints'], pred['keypoint_scores'])
                ])
                
                normalized_features = self._preprocess_keypoints(kps_raw)
                if normalized_features is not None:
                    features_list.append(normalized_features)
            
            return features_list
            
        except Exception as e:
            print(f"Error in MMPose keypoint extraction: {e}")
            return self._create_mock_keypoints()

    def _create_mock_keypoints(self):
        """Create mock keypoints for testing when MMPose is not available."""
        # Create a single mock person with valid keypoints
        mock_features = np.random.randn(51).astype(np.float32) * 0.1
        return [mock_features]

    def execute(self, requests):
        """Execute preprocessing for a batch of requests."""
        responses = []
        
        for request in requests:
            # Get input image data
            input_image = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            image_data = input_image.as_numpy()
            
            batch_features = []
            
            # Process each image in the batch
            for img_bytes in image_data:
                try:
                    # Decode image from bytes
                    if isinstance(img_bytes[0], bytes):
                        # Handle byte string input
                        nparr = np.frombuffer(img_bytes[0], np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    else:
                        # Handle numpy array input
                        image = img_bytes.astype(np.uint8)
                    
                    if image is None:
                        print("Failed to decode image")
                        # Use mock features for failed decoding
                        features_list = self._create_mock_keypoints()
                    else:
                        # Extract keypoints and normalize
                        features_list = self._extract_keypoints_mmpose(image)
                    
                    # Handle multiple people or no people detected
                    if len(features_list) == 0:
                        # No people detected - use zero features
                        features = np.zeros(51, dtype=np.float32)
                    elif len(features_list) == 1:
                        # Single person detected
                        features = features_list[0]
                    else:
                        # Multiple people - for now, take the first person
                        # In production, you might want to handle this differently
                        features = features_list[0]
                    
                    batch_features.append(features)
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
                    # Use zero features as fallback
                    features = np.zeros(51, dtype=np.float32)
                    batch_features.append(features)
            
            # Convert to output tensor
            output_features = np.array(batch_features, dtype=self.output_dtype)
            output_tensor = pb_utils.Tensor("NORMALIZED_FEATURES", output_features)
            
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        
        return responses

    def finalize(self):
        """Clean up resources."""
        print('Preprocessing model done')