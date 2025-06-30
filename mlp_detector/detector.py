import os
import cv2
import numpy as np
import torch
from mmpose.apis import MMPoseInferencer
from .mlp_model import MLPModel 

MODEL_WEIGHTS_PATH = 'mlp_phone_detector_best.pth' 
INPUT_DIM = 51 
HIDDEN_DIMS = [128, 64]
OUTPUT_DIM = 2 

# Initialize MMPose Inferencer once
_inferencer = MMPoseInferencer('human', device='cpu')

# Keypoint indices for COCO-17 format
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
R_SHOULDER = 6
L_HIP, R_HIP = 11, 12

# Helper functions
def _get_kp_coords(kps, idx):
    kp = kps[idx]
    return kp[:2] if kp[2] > 0.3 else None

def _dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def _preprocess_keypoints(kps_raw):
    l_shoulder = _get_kp_coords(kps_raw, L_SHOULDER)
    r_shoulder = _get_kp_coords(kps_raw, R_SHOULDER)
    l_hip = _get_kp_coords(kps_raw, L_HIP)
    r_hip = _get_kp_coords(kps_raw, R_HIP)

    neck_proxy = None
    if l_shoulder is not None and r_shoulder is not None:
        neck_proxy = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
    
    body_scale = 0
    if l_shoulder is not None and r_shoulder is not None:
        body_scale = _dist(l_shoulder, r_shoulder)
    elif l_hip is not None and r_hip is not None:
        body_scale = _dist(l_hip, r_hip) 
    
    if neck_proxy is None or body_scale < 1e-6: 
        return None 

    normalized_kps = []
    for i in range(kps_raw.shape[0]): 
        kp_coords = _get_kp_coords(kps_raw, i)
        kp_score = kps_raw[i, 2]

        if kp_coords is not None:
            centered_kp = np.array(kp_coords) - np.array(neck_proxy)
            scaled_kp = centered_kp / body_scale
            normalized_kps.extend(scaled_kp.tolist()) 
        else:
            normalized_kps.extend([0.0, 0.0]) 
        normalized_kps.append(kp_score) 
    
    if len(normalized_kps) == INPUT_DIM:
        return np.array(normalized_kps, dtype=np.float32)
    else:
        return None 


def load_mlp_model(model_path: str, device: torch.device) -> MLPModel:
    model = MLPModel(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, output_dim=OUTPUT_DIM).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}. Please train the model first.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_phone_usage(image_path: str, model: MLPModel, device: torch.device):
    results = []

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}.")
        return results

    result_generator = _inferencer(image_path, return_vis=False)
    result = next(result_generator)
    predictions = result['predictions'][0]

    if not predictions:
        return results 

    model.eval() 
    with torch.no_grad():
        for i, pred in enumerate(predictions):
            kps_raw = np.array([(kp[0], kp[1], score) for kp, score in zip(pred['keypoints'], pred['keypoint_scores'])])
            
            normalized_features = _preprocess_keypoints(kps_raw)
            
            if normalized_features is None:
                continue

            input_tensor = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0).to(device)
            
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0] 
            
            phone_prob = probabilities[1].item() 
            
            is_phone = phone_prob > 0.5 

            results.append({
                'person_id': i,
                'is_phone': is_phone,
                'confidence': phone_prob
            })

    return results


def detect_and_annotate_phone_in_image(image_path: str, output_path: str, model: MLPModel, device: torch.device):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}.")
        return False

    annotated_img = img.copy()
    phone_detected_in_image = False

    result_generator = _inferencer(image_path, return_vis=False)
    result = next(result_generator)
    predictions = result['predictions'][0]

    KEYPOINT_COLORS = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (0, 128, 255), (128, 0, 255), (255, 128, 0), (0, 255, 128), (128, 255, 0), (255, 0, 128),
        (0, 128, 128), (128, 0, 128), (128, 128, 0), (64, 128, 0), (0, 64, 128)
    ]
    KEYPOINT_RADIUS = 5
    KEYPOINT_THICKNESS = -1

    model.eval()
    with torch.no_grad():
        for pred in predictions:
            kps_raw = np.array([(kp[0], kp[1], score) for kp, score in zip(pred['keypoints'], pred['keypoint_scores'])])
            
            normalized_features = _preprocess_keypoints(kps_raw)
            
            if normalized_features is None:
                continue

            input_tensor = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            phone_prob = probabilities[1].item()
            is_phone = phone_prob > 0.5

            # Draw keypoints
            for i, (kp_coords, kp_score) in enumerate(zip(pred['keypoints'], pred['keypoint_scores'])):
                if kp_score > 0.3:
                    color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
                    cv2.circle(annotated_img, (int(kp_coords[0]), int(kp_coords[1])), KEYPOINT_RADIUS, color, KEYPOINT_THICKNESS)

            # Draw bounding box and label if phone detected
            if is_phone:
                phone_detected_in_image = True
                bbox = pred['bbox'][0]
                x1, y1, x2, y2 = map(int, bbox)
                label = f"Phone: {phone_prob:.2f}"
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(output_path, annotated_img)
    return phone_detected_in_image