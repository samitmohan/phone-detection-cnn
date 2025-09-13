
import numpy as np
import requests
import json
import cv2
from PIL import Image

TRITON_URL = "http://localhost:8000"

def triton_to_numpy_dtype(triton_dtype):
    if triton_dtype == "BOOL":
        return np.bool_
    elif triton_dtype == "UINT8":
        return np.uint8
    elif triton_dtype == "UINT16":
        return np.uint16
    elif triton_dtype == "UINT32":
        return np.uint32
    elif triton_dtype == "UINT64":
        return np.uint64
    elif triton_dtype == "INT8":
        return np.int8
    elif triton_dtype == "INT16":
        return np.int16
    elif triton_dtype == "INT32":
        return np.int32
    elif triton_dtype == "INT64":
        return np.int64
    elif triton_dtype == "FP16":
        return np.float16
    elif triton_dtype == "FP32":
        return np.float32
    elif triton_dtype == "FP64":
        return np.float64
    elif triton_dtype == "BYTES":
        return np.object_
    return None

def get_model_metadata(model_name):
    url = f"{TRITON_URL}/v2/models/{model_name}"
    response = requests.get(url)
    return response.json()

def get_input_metadata(model_metadata, input_name):
    for inp in model_metadata['inputs']:
        if inp['name'] == input_name:
            return inp
    return None

def get_output_metadata(model_metadata, output_name):
    for outp in model_metadata['outputs']:
        if outp['name'] == output_name:
            return outp
    return None

def infer(model_name, input_data, input_names, output_names):
    url = f"{TRITON_URL}/v2/models/{model_name}/infer"

    inputs = []
    for i, data in enumerate(input_data):
        model_metadata = get_model_metadata(model_name)
        input_metadata = get_input_metadata(model_metadata, input_names[i])

        if input_metadata['datatype'] == 'UINT8':
            inputs.append({
                "name": input_names[i],
                "shape": [1, len(data)],
                "datatype": "UINT8",
                "data": list(data)
            })
        else:
            inputs.append({
                "name": input_names[i],
                "shape": data.shape,
                "datatype": input_metadata['datatype'],
                "data": data.tolist()
            })

    payload = {
        "inputs": inputs,
        "outputs": [{ "name": name } for name in output_names]
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()

    outputs = {}
    for output_metadata in result['outputs']:
        output_name = output_metadata['name']
        output_shape = output_metadata['shape']
        triton_dtype = output_metadata['datatype']
        numpy_dtype = triton_to_numpy_dtype(triton_dtype)
        output_data = np.array(output_metadata['data'], dtype=numpy_dtype).reshape(output_shape)
        outputs[output_name] = output_data

    return outputs

def preprocess_image(image_path):
    with open(image_path, 'rb') as f:
        return f.read()

def save_tensor_as_image(tensor, filename):
    img_chw = tensor.squeeze(0)
    img_hwc = np.transpose(img_chw, (1, 2, 0))
    img_uint8 = (img_hwc * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"/tmp/{filename}", img_bgr)

def draw_detections(image_tensor, dets, labels, filename):
    img_chw = image_tensor.squeeze(0)
    img_hwc = np.transpose(img_chw, (1, 2, 0))
    img_uint8 = (img_hwc * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    for i in range(len(dets[0])):
        score = dets[0][i][4]
        if score > 0.3:
            label = int(labels[0][i])
            x1, y1, x2, y2 = dets[0][i][:4].astype(int)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"Label: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite(f"/tmp/{filename}", img_bgr)

def draw_pose(image_tensor, simcc_x, simcc_y, filename):
    img_chw = image_tensor.squeeze(0)
    img_hwc = np.transpose(img_chw, (1, 2, 0))
    img_uint8 = (img_hwc * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    keypoints_x = np.argmax(simcc_x.squeeze(0), axis=1)
    keypoints_y = np.argmax(simcc_y.squeeze(0), axis=1)

    for i in range(len(keypoints_x)):
        x = int(keypoints_x[i] * img_bgr.shape[1] / simcc_x.shape[2])
        y = int(keypoints_y[i] * img_bgr.shape[0] / simcc_y.shape[2])
        cv2.circle(img_bgr, (x, y), 5, (0, 0, 255), -1)

    cv2.imwrite(f"/tmp/{filename}", img_bgr)

def debug_pipeline(image_path):
    print(f"--- Debugging pipeline for {image_path} ---")
    image_name = image_path.split("/")[-1].split(".")[0]

    # 1. Preprocess image
    print("1. Preprocessing image...")
    preprocessed_image_data = preprocess_image(image_path)
    preprocessed_output = infer(
        "preprocess_img",
        [preprocessed_image_data],
        ["INPUT_IMAGE"],
        ["PROCESSED_IMAGE"]
    )
    processed_image = preprocessed_output["PROCESSED_IMAGE"]
    print(f"  - PROCESSED_IMAGE shape: {processed_image.shape}")
    save_tensor_as_image(processed_image, f"{image_name}_1_preprocessed.jpg")
    print(f"  - Saved visualization to /tmp/{image_name}_1_preprocessed.jpg")

    # 2. RTMDet detection
    print("\n2. RTMDet detection...")
    rtmdet_output = infer(
        "rtmdet_detection",
        [processed_image],
        ["input"],
        ["dets", "labels"]
    )
    dets = rtmdet_output["dets"]
    labels = rtmdet_output["labels"]
    print(f"  - dets shape: {dets.shape}")
    print(f"  - labels shape: {labels.shape}")
    draw_detections(processed_image, dets, labels, f"{image_name}_2_detections.jpg")
    print(f"  - Saved visualization to /tmp/{image_name}_2_detections.jpg")

    # 3. Person cropper
    print("\n3. Person cropper...")
    person_cropper_output = infer(
        "person_cropper",
        [processed_image, dets, labels],
        ["image", "dets", "labels"],
        ["cropped_person"]
    )
    cropped_person = person_cropper_output["cropped_person"]
    print(f"  - cropped_person shape: {cropped_person.shape}")
    save_tensor_as_image(cropped_person, f"{image_name}_3_cropped_person.jpg")
    print(f"  - Saved visualization to /tmp/{image_name}_3_cropped_person.jpg")

    # 4. RTMPose estimation
    print("\n4. RTMPose estimation...")
    rtmpose_output = infer(
        "rtmpose_estimation",
        [cropped_person],
        ["input"],
        ["simcc_x", "simcc_y"]
    )
    simcc_x = rtmpose_output["simcc_x"]
    simcc_y = rtmpose_output["simcc_y"]
    print(f"  - simcc_x shape: {simcc_x.shape}")
    print(f"  - simcc_y shape: {simcc_y.shape}")
    draw_pose(cropped_person, simcc_x, simcc_y, f"{image_name}_4_pose.jpg")
    print(f"  - Saved visualization to /tmp/{image_name}_4_pose.jpg")

    # 5. Feature normalizer
    print("\n5. Feature normalizer...")
    feature_normalizer_output = infer(
        "feature_normalizer",
        [simcc_x, simcc_y],
        ["simcc_x", "simcc_y"],
        ["normalized_features"]
    )
    normalized_features = feature_normalizer_output["normalized_features"]
    print(f"  - normalized_features shape: {normalized_features.shape}")
    print("DEBUG: Normalized Features Array:")
    print(normalized_features)

    # 6. MLP phone detector
    print("\n6. MLP phone detector...")
    mlp_output = infer(
        "mlp_phone_detector",
        [normalized_features],
        ["input"],
        ["output"]
    )
    mlp_logits = mlp_output["output"]
    print(f"  - mlp_logits: {mlp_logits}")

    # 7. Postprocess
    print("\n7. Postprocess...")
    postprocess_output = infer(
        "postprocess",
        [mlp_logits],
        ["MODEL_OUTPUT"],
        ["PERSON_ID", "IS_PHONE", "CONFIDENCE"]
    )
    person_id = postprocess_output["PERSON_ID"]
    is_phone = postprocess_output["IS_PHONE"]
    confidence = postprocess_output["CONFIDENCE"]
    print(f"  - PERSON_ID: {person_id}")
    print(f"  - IS_PHONE: {bool(is_phone)}")
    print(f"  - CONFIDENCE: {confidence}")

if __name__ == '__main__':
    phone_image_path = '/app/deployment/test_images/phone6.png'
    nophone_image_path = '/app/deployment/test_images/nophone.jpg'

    debug_pipeline(phone_image_path)
    debug_pipeline(nophone_image_path)
