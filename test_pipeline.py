#!/usr/bin/env python3
import numpy as np
import cv2
import requests
import json
import time
import os
import sys

def test_pipeline_accuracy():
    """Test the 5-stage pipeline on real phone/no-phone images"""

    # Test cases: (filename, expected_phone, description)
    test_cases = [
        ('phone_img.jpg', True, 'Person with phone'),
        ('phone.png', True, 'Phone image'),
        ('phone2.png', True, 'Phone image 2'),
        ('phone3.png', True, 'Phone image 3'),
        ('nophone.jpg', False, 'No phone image'),
        ('nophone2.jpg', False, 'No phone image 2'),
        ('nophone3.jpg', False, 'No phone image 3'),
    ]

    correct_predictions = 0
    total_predictions = 0
    results = []

    for filename, expected_phone, description in test_cases:
        print(f'Testing: {filename} ({description})')

        image_path = f'/app/deployment/test_images/{filename}'

        try:
            # Read image
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img_bytes = f.read()
            else:
                print(f'Image not found: {image_path}')
                continue

            print(f'Image size: {len(img_bytes)} bytes')

            # Send to pipeline
            url = 'http://localhost:8000/v2/models/ensemble_phone_detection/infer'
            data = {
                'inputs': [
                    {
                        'name': 'INPUT_IMAGE',
                        'datatype': 'UINT8',
                        'shape': [1, len(img_bytes)],
                        'data': list(img_bytes)
                    }
                ]
            }

            start_time = time.time()
            response = requests.post(url, json=data, timeout=120)
            inference_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()

                # Parse results
                phone_detected = False
                confidence = 0.0

                for output in result.get('outputs', []):
                    name = output.get('name')
                    data_vals = output.get('data', [])

                    if name == 'IS_PHONE' and data_vals:
                        phone_detected = bool(data_vals[0])
                    elif name == 'CONFIDENCE' and data_vals:
                        confidence = data_vals[0]

                # Check accuracy
                is_correct = (phone_detected == expected_phone)
                correct_predictions += int(is_correct)
                total_predictions += 1

                status_icon = '✅' if is_correct else '❌'
                prediction_text = 'PHONE' if phone_detected else 'NO PHONE'
                expected_text = 'PHONE' if expected_phone else 'NO PHONE'

                print(f'   {status_icon} Result: {prediction_text} (conf: {confidence:.3f}) | Expected: {expected_text}')
                print(f'Time: {inference_time:.1f}ms')

                results.append({
                    'filename': filename,
                    'expected': expected_phone,
                    'predicted': phone_detected,
                    'confidence': confidence,
                    'correct': is_correct,
                    'time': inference_time
                })

            else:
                print(f'Pipeline error: {response.status_code}')
                print(f'      {response.text[:200]}')

        except Exception as e:
            print(f'Test failed: {e}')


    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        avg_time = np.mean([r['time'] for r in results]) if results else 0

        print(f'Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.1f}%')
        print(f'Average inference time: {avg_time:.1f}ms')
        print()

        if accuracy >= 80:
            print('EXCELLENT: Pipeline accuracy is good!')
        elif accuracy >= 60:
            print('MODERATE: Some accuracy issues detected')
        else:
            print('POOR: Significant accuracy problems')
            print('RTMDet model may need replacement from /tmp weights')

        for r in results:
            status = '✅' if r['correct'] else '❌'
            pred_text = 'PHONE' if r['predicted'] else 'NO_PHONE'
            print(f'   {status} {r["filename"]:15} → {pred_text:8} (conf: {r["confidence"]:.3f})')

    else:
        print('No successful tests completed')
    return results

if __name__ == '__main__':
    results = test_pipeline_accuracy()
