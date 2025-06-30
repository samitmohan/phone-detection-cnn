# MLP Phone Detector

A keypoint-based phone detection system leveraging a Multi-Layer Perceptron (MLP) and HRNet pose estimation.

## Overview

This package provides a command-line tool to detect if a person in an image is using a phone. Unlike rule-based systems, this detector uses a trained MLP model that learns complex patterns directly from normalized human pose keypoints extracted by HRNet.

## Features

*   **Pose Estimation:** Utilizes the HRNet model via the MMPose library to accurately detect human keypoints.
*   **MLP-Based Classification:** Employs a trained Multi-Layer Perceptron to classify normalized keypoint data as "phone usage" or "no phone usage."
*   **Data Normalization:** Keypoints are normalized to be invariant to person size and position, ensuring robust model performance.
*   **Simple CLI:** Easy-to-use command-line interface for quick detection.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/samitmohan//phone-detection-cnn.git
    cd mlp-phone-detector
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv_mlp
    source venv_mlp/bin/activate  # On Windows, use `venv_mlp\Scripts\activate`
    ```

3.  **Install the package:**
    ```bash
    pip install .
    ```
    This will install all necessary dependencies, including `torch` and `mmpose`.

## Usage

To use this detector, you first need to train the MLP model.

### Step 1: Prepare Your Data (for Training)

This step extracts keypoints from your images, normalizes them, and saves them along with their ground truth labels. This creates the dataset for training the MLP.

**Prerequisites:**

*   Your raw images should be located in a directory (e.g., `my_images/`).
*   You need an `annotations.csv` file (e.g., `my_annotations.csv`) containing ground truth labels for these images. The format should be `image_path,is_phone_usage`.

**Run the data collector:**

```bash
python data_collector.py
```

This will generate two files in the current directory:

*   `keypoints_data.npy`: A NumPy array containing the 51-element normalized keypoint feature vectors for each person detected.
*   `labels.npy`: A NumPy array containing the corresponding binary labels (0 for no phone, 1 for phone).

Have attached both these based on the images I have trained. You can retrain them on more images and generate these files again.

### Step 2: Train the MLP Model

This step trains the MLP using the data prepared in Step 1.

**Run the training script:**

```bash
python train_mlp.py
```

This script will:

*   Load `keypoints_data.npy` and `labels.npy`.
*   Split the data into training and validation sets.
*   Initialize and train the `MLPModel`.
*   Save the best performing model weights as `mlp_phone_detector_best.pth`.
*   Print training progress and final evaluation metrics.

### Step 3: Use the Trained Model for Inference

Once the model is trained and `mlp_phone_detector_best.pth` is available in the root of your package, you can use the `detect-phone-mlp` command from your terminal.

### Basic Detection

To simply check if phone usage is detected in one or more images:

```bash
detect-phone-mlp --image path/to/image1.jpg path/to/image2.jpg
```


## Example:  

Person 0: using a phone (Confidence: 0.9525)

OR

Person 0: NOT using a phone (Confidence: 0.4750)

```bash
detect-phone-mlp --image my_test_images/person_on_phone.jpg
```

### Detection with Annotated Output

To save annotated versions of the images (with keypoints, bounding boxes, and confidence scores):

```bash
detect-phone-mlp --image path/to/image1.jpg --output output_directory/
```

Example:
```bash
detect-phone-mlp --image my_test_images/person_on_phone.jpg --output output/detected_phone.jpg
```

## Important Notes

*   **MMPose Model Download:** The first time you run any script that uses `MMPoseInferencer` (e.g., `data_collector.py` or `detect-phone-mlp`), `mmpose` will automatically download the pre-trained HRNet model weights (typically `human` model, COCO-trained). This might take some time depending on your internet connection.
*   **COCO Weights Path:** The HRNet model used is pre-trained on the COCO dataset. The weights are managed by the `mmpose` library and are downloaded to a default cache location (e.g., `~/.cache/openmmlab/mmpose/checkpoints/`). You generally do not need to manage these weights manually.
*   **Performance:** The accuracy of the detection depends heavily on the quality and diversity of your training data, as well as the accuracy of the underlying pose estimation. This MLP-based system is designed to be more robust than rule-based approaches but still relies on good quality keypoint data.
*   **CPU vs. GPU:** By default, the detector runs on the CPU. For faster inference and training, ensure you have a CUDA-enabled GPU and the appropriate PyTorch version installed. The `mmpose` library will automatically try to use the GPU if available.
