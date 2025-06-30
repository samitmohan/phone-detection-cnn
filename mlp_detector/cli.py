import argparse
import os
import torch
from . import load_mlp_model, predict_phone_usage, detect_and_annotate_phone_in_image

MODEL_WEIGHTS_PATH = 'mlp_phone_detector_best.pth' 

def main():
    parser = argparse.ArgumentParser(description="Detect phone usage in images using an MLP model.")
    parser.add_argument("--image", type=str, nargs='+', required=True, help="Path(s) to the input image file(s).")
    parser.add_argument("--output", type=str, help="Optional: Directory to save annotated output images.")
    parser.add_argument("--model_path", type=str, default=MODEL_WEIGHTS_PATH, help="Path to the trained MLP model weights (.pth file).")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        model = load_mlp_model(args.model_path, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have trained the model and 'mlp_phone_detector_best.pth' is in the correct location.")
        return

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    for img_path in args.image:
        if not os.path.exists(img_path):
            print(f"Error: Image file not found at {img_path}. Skipping.")
            continue

        print(f"Processing image: {img_path}")
        
        if args.output:
            output_filename = os.path.basename(img_path)
            output_full_path = os.path.join(args.output, output_filename)
            is_phone_detected = detect_and_annotate_phone_in_image(img_path, output_full_path, model, device)
            if is_phone_detected:
                print(f"  Phone usage detected. Annotated image saved to {output_full_path}")
            else:
                print(f"  No phone usage detected. Annotated image saved to {output_full_path}")
        else:
            predictions = predict_phone_usage(img_path, model, device)
            if predictions:
                for pred in predictions:
                    status = "using a phone" if pred['is_phone'] else "NOT using a phone"
                    print(f"  Person {pred['person_id']}: {status} (Confidence: {pred['confidence']:.4f})")
            else:
                print("  No people detected or no valid predictions made in this image.")

if __name__ == "__main__":
    main()