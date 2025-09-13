import triton_python_backend_utils as pb_utils
import numpy as np
import json

class TritonPythonModel:
    """
    Feature normalizer that EXACTLY matches the normalization logic used in data_collector.py
    for training the MLP model. This ensures the MLP receives the same 51-dimensional feature
    vectors it was trained on.
    """

    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        # No MMPose dependencies - use simple argmax decoding like training

    def _get_kp_coords(self, kps, idx):
        """Returns keypoint coordinates if confidence is above threshold - matches data_collector.py"""
        kp = kps[idx]
        return kp[:2] if kp[2] > 0.3 else None

    def _dist(self, p1, p2):
        """Calculate distance between two points - matches data_collector.py"""
        return np.linalg.norm(p1 - p2)

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get the input tensors (simcc_x and simcc_y)
            in_x = pb_utils.get_input_tensor_by_name(request, 'simcc_x')
            in_y = pb_utils.get_input_tensor_by_name(request, 'simcc_y')
            simcc_x = in_x.as_numpy().squeeze(0)  # Remove batch dimension 
            simcc_y = in_y.as_numpy().squeeze(0)  # Remove batch dimension

            # Decode simcc heatmaps to keypoints using simple argmax (like MMPose internally does)
            # simcc_x shape: (17, 384), simcc_y shape: (17, 512)
            keypoint_x = np.argmax(simcc_x, axis=1)  # (17,) - x coordinates
            keypoint_y = np.argmax(simcc_y, axis=1)  # (17,) - y coordinates
            
            # Get confidence scores (max values from heatmaps)
            confidence_x = np.max(simcc_x, axis=1)  # (17,)
            confidence_y = np.max(simcc_y, axis=1)  # (17,)
            keypoint_scores = (confidence_x + confidence_y) / 2.0  # Average confidence
            
            # Scale keypoints to image coordinates (RTMPose uses 256x192 input)
            # simcc_x maps to 192 width, simcc_y maps to 256 height  
            keypoint_x_scaled = keypoint_x * (192.0 / 384.0)  # Scale to 192 width
            keypoint_y_scaled = keypoint_y * (256.0 / 512.0)  # Scale to 256 height
            
            # Create keypoints array in same format as training: (17, 3) with [x, y, score]
            kps_raw = np.zeros((17, 3), dtype=np.float32)
            for i in range(17):
                kps_raw[i] = [keypoint_x_scaled[i], keypoint_y_scaled[i], keypoint_scores[i]]

            # --- EXACT LOGIC FROM data_collector.py LINES 71-104 ---
            
            # Define keypoint indices (matches data_collector.py lines 17-20)
            L_SHOULDER, R_SHOULDER = 5, 6
            L_HIP, R_HIP = 11, 12

            # Get keypoint coordinates using same confidence threshold (0.3)
            l_shoulder = self._get_kp_coords(kps_raw, L_SHOULDER)
            r_shoulder = self._get_kp_coords(kps_raw, R_SHOULDER)
            l_hip = self._get_kp_coords(kps_raw, L_HIP)
            r_hip = self._get_kp_coords(kps_raw, R_HIP)

            # Calculate neck proxy (midpoint of shoulders) - line 77-79
            neck_proxy = None
            if l_shoulder is not None and r_shoulder is not None:
                neck_proxy = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
            
            # Calculate body scale (distance between shoulders or shoulder-hip) - lines 82-86
            body_scale = 0
            if l_shoulder is not None and r_shoulder is not None:
                body_scale = self._dist(l_shoulder, r_shoulder)
            elif l_hip is not None and r_hip is not None:
                body_scale = self._dist(l_hip, r_hip)
            
            # Check for insufficient keypoints - lines 88-90
            if neck_proxy is None or body_scale < 1e-6:
                # Output zero features if normalization not possible
                normalized_features = np.zeros(51, dtype=np.float32)
            else:
                # Normalize all 17 keypoints - lines 92-104
                normalized_kps = []
                for i in range(17):  # All 17 keypoints
                    kp_coords = self._get_kp_coords(kps_raw, i)
                    kp_score = kps_raw[i, 2]

                    if kp_coords is not None:
                        # Center around neck and scale by body_scale - lines 99-100
                        centered_kp = np.array(kp_coords) - np.array(neck_proxy)
                        scaled_kp = centered_kp / body_scale
                        normalized_kps.extend(scaled_kp.tolist())  # Add x, y
                    else:
                        normalized_kps.extend([0.0, 0.0])  # Zero for missing keypoints
                    normalized_kps.append(kp_score)  # Add confidence score
                
                # Convert to numpy array - should be exactly 51 elements
                normalized_features = np.array(normalized_kps, dtype=np.float32)
            
            # --- END EXACT LOGIC FROM data_collector.py ---

            # Ensure correct shape and create output tensor
            normalized_features = normalized_features.reshape(1, -1)  # Shape: (1, 51)
            out_tensor = pb_utils.Tensor('normalized_features', normalized_features)
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
        return responses