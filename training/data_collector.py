import os
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
import glob
import csv
from tqdm import tqdm

IMAGES_DIR = '../HRNetPoseModel/images_smoothed' 
ANNOTATIONS_FILE = '../HRNetPoseModel/annotations.csv' 
OUTPUT_DIR = '.' 
KEYPOINTS_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'keypoints_data.npy')
LABELS_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'labels.npy')

_inferencer = MMPoseInferencer('human', device='cpu')

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
R_SHOULDER = 6
L_HIP, R_HIP = 11, 12

def _get_kp_coords(kps, idx):
    """Returns keypoint coordinates if confidence is above threshold."""
    kp = kps[idx]
    return kp[:2] if kp[2] > 0.3 else None

def _dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def collect_and_normalize_keypoints():
    all_keypoints_data = []
    all_labels = []

    ground_truth_map = {}
    try:
        with open(ANNOTATIONS_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_basename = os.path.basename(row['image_path'])
                ground_truth_map[image_basename] = int(row['is_phone_usage'])
    except FileNotFoundError:
        print(f"Error: Annotations file '{ANNOTATIONS_FILE}' not found.")
        return

    image_files = [f for f in glob.glob(os.path.join(IMAGES_DIR, '**', '*.*'), recursive=True) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to process.")

    for img_path in tqdm(image_files, desc="Collecting Keypoint Data"):
        image_basename = os.path.basename(img_path)
        if image_basename not in ground_truth_map:
            print(f"Warning: '{image_basename}' not found in annotations. Skipping.")
            continue

        true_label = ground_truth_map[image_basename]

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        # pose inference
        result_generator = _inferencer(img_path, return_vis=False)
        result = next(result_generator)
        predictions = result['predictions'][0]

        for pred in predictions:
            kps_raw = np.array([(kp[0], kp[1], score) for kp, score in zip(pred['keypoints'], pred['keypoint_scores'])])
            
            # normalisation (keypts)
            l_shoulder = _get_kp_coords(kps_raw, L_SHOULDER)
            r_shoulder = _get_kp_coords(kps_raw, R_SHOULDER)
            l_hip = _get_kp_coords(kps_raw, L_HIP)
            r_hip = _get_kp_coords(kps_raw, R_HIP)

            # neck proxy (midpoint of shoulders)
            neck_proxy = None
            if l_shoulder is not None and r_shoulder is not None:
                neck_proxy = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
            
            # body scale (distance between shoulders or shoulder-hip if shoulders not available)
            body_scale = 0
            if l_shoulder is not None and r_shoulder is not None:
                body_scale = _dist(l_shoulder, r_shoulder)
            elif l_hip is not None and r_hip is not None:
                body_scale = _dist(l_hip, r_hip) 
            
            if neck_proxy is None or body_scale < 1e-6:
                # print(f"Skipping person in {image_basename} due to insufficient keypoints for normalization.")
                continue

            normalized_kps = []
            for i in range(kps_raw.shape[0]): # iterate through all 17 keypoints
                kp_coords = _get_kp_coords(kps_raw, i)
                kp_score = kps_raw[i, 2]

                if kp_coords is not None:
                    # Center and scale
                    centered_kp = np.array(kp_coords) - np.array(neck_proxy)
                    scaled_kp = centered_kp / body_scale
                    normalized_kps.extend(scaled_kp.tolist()) # Add x, y
                else:
                    normalized_kps.extend([0.0, 0.0]) 
                normalized_kps.append(kp_score) 
            
            # Ensure the feature vector size (51)
            if len(normalized_kps) == 51:
                all_keypoints_data.append(normalized_kps)
                all_labels.append(true_label)
            else:
                print(f"Warning: Skipping person in {image_basename} due to incorrect feature vector length ({len(normalized_kps)}).")

    # conv lists to np arrays
    final_keypoints_data = np.array(all_keypoints_data, dtype=np.float32)
    final_labels = np.array(all_labels, dtype=np.int32)

    np.save(KEYPOINTS_OUTPUT_FILE, final_keypoints_data)
    np.save(LABELS_OUTPUT_FILE, final_labels)

    print("Data collection complete.")
    print(f"Saved {len(final_keypoints_data)} keypoint samples to '{KEYPOINTS_OUTPUT_FILE}'.")
    print(f"Saved {len(final_labels)} labels to '{LABELS_OUTPUT_FILE}'.")

if __name__ == '__main__':
    collect_and_normalize_keypoints()
