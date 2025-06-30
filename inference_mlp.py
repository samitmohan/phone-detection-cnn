import os
import cv2
import numpy as np
import torch
from mmpose.apis import MMPoseInferencer
from mlp_model import MLPModel

MODEL_WEIGHTS_PATH = 'mlp_phone_detector_best.pth'
INPUT_DIM = 51 
HIDDEN_DIMS = [128, 64]
OUTPUT_DIM = 2 

_inferencer = MMPoseInferencer('human', device='cpu')

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
R_SHOULDER = 6
L_HIP, R_HIP = 11, 12

def _get_kp_coords(kps, idx):
    kp = kps[idx]
    return kp[:2] if kp[2] > 0.3 else None

def _dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def preprocess_keypoints(kps_raw):
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
    for i in range(kps_raw.shape[0]): # Iterate through all 17 keypoints
        kp_coords = _get_kp_coords(kps_raw, i)
        kp_score = kps_raw[i, 2]

        if kp_coords is not None:
            # Center and scale
            # should add smoothning here ? 
            centered_kp = np.array(kp_coords) - np.array(neck_proxy)
            scaled_kp = centered_kp / body_scale
            normalized_kps.extend(scaled_kp.tolist()) # add x, y
        else:
            normalized_kps.extend([0.0, 0.0])
        normalized_kps.append(kp_score)
    
    if len(normalized_kps) == INPUT_DIM:
        return np.array(normalized_kps, dtype=np.float32)
    else:
        return None 


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
        return results # No people detected

    model.eval()
    with torch.no_grad():
        for i, pred in enumerate(predictions):
            kps_raw = np.array([(kp[0], kp[1], score) for kp, score in zip(pred['keypoints'], pred['keypoint_scores'])])
            
            normalized_features = preprocess_keypoints(kps_raw)
            
            if normalized_features is None:
                # print(f"Skipping person {i} in {image_path} due to insufficient keypoints for normalization.")
                continue

            # Convert to tensor and add batch dimension
            input_tensor = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0).to(device)
            
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0] 
            
            phone_prob = probabilities[1].item() # prob of being phone class (assuming 0=nophone, 1=phone)
            
            is_phone = phone_prob > 0.5 

            results.append({
                'person_id': i,
                'is_phone': is_phone,
                'confidence': phone_prob
            })

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Predict phone usage in an image using a trained MLP model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--model_path", type=str, default=MODEL_WEIGHTS_PATH, help="Path to the trained MLP model weights (.pth file).")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        exit()
    if not os.path.exists(args.model_path):
        print(f"Error: Model weights not found at {args.model_path}. Please train the model first.")
        exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MLPModel(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, output_dim=OUTPUT_DIM).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval() 

    print(f"\nProcessing image: {args.image}")
    predictions = predict_phone_usage(args.image, model, device)

    if predictions:
        for pred in predictions:
            status = "using a phone" if pred['is_phone'] else "NOT using a phone"
            print(f"Person {pred['person_id']}: {status} (Confidence: {pred['confidence']:.4f})")
    else:
        print("No people detected or no valid predictions made in the image.")
