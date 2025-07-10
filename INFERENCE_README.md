# Phone Usage Detection - Inference Guide

The system works by:
1. Detecting human poses in images using MMPose
2. Extracting and normalizing keypoint features
3. Using a trained MLP model to classify if a person is using a phone

## Prerequisites

- Python 3.8 or higher
- PyTorch
- OpenCV
- MMPose and its dependencies
- Trained model weights file (`mlp_phone_detector_best.pth`)

## Installation
```bash
python3 -m venv .venv           
source .venv/bin/activate       
```

1. Install required files: (After creating new environment)
```bash
pip3 install -r requirements.txt # downloads all requirements
```

3. Install the project package in development mode:
```bash
pip install .
```

This will install the package in editable mode, making the `mlp_detector` module available in your Python environment.

## Usage

After installation, you can use the command-line tool `detect-phone-mlp` to analyze images:

```bash
detect-phone-mlp --image <path_to_image> [--model_path <path_to_weights>]
```

### Arguments:
- `--image`: (Required) Path to the input image file
- `--model_path`: (Optional) Path to the trained model weights. Defaults to 'mlp_phone_detector_best.pth'

### Example:
```bash
detect-phone-mlp --image /path/to/your/image.jpg
```

### Output Format:

```

Using device: cpu
Processing image: examples/test.jpg
Person 0: using a phone (Confidence: 0.8532)
Person 1: NOT using a phone (Confidence: 0.1245)
The script will output predictions for each person detected in the image:

Actual Output:

>>> detect-phone-mlp --image /Users/smol/office_work/phone-detection/HRNetPoseModel/images_smoothed/phone2.png  
Loads checkpoint by http backend from path: https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth
 Person 0: using a phone (Confidence: 0.9525)
```

## Understanding the Results

- Each detected person gets an ID (starting from 0)
- For each person, the system provides:
  - Phone usage status (using/NOT using)
  - Confidence score (0-1, higher means more confident)
- A confidence threshold of 0.5 is used to determine phone usage

## Notes
2. **Common Issues**:
   - If no predictions are made, it could mean:
     - No people were detected in the image
     - Detected poses had insufficient keypoint quality
     - Key body parts (shoulders/neck) were not visible

3. **Performance Considerations**:
   - Image quality affects pose detection accuracy
   - Multiple people in an image will increase processing time
   - GPU acceleration significantly improves performance

## Errors

1. If MMCV/MMPose related errors:
   ```bash
   pip uninstall mmcv mmcv-full -y
   pip install "mmcv>=2.0.0rc4,<2.2.0"
   ```

2. If model weights are not found:
   - Make sure `mlp_phone_detector_best.pth` is in the current directory
   - Or specify the correct path using `--model_path`
