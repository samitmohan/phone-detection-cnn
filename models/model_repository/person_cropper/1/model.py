import json
import numpy as np
import triton_python_backend_utils as pb_utils
import cv2


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        
        # Get the model configuration
        input_configs = self.model_config['input']
        output_configs = self.model_config['output']
        
        # Store input/output names
        self.input_names = {}
        self.output_names = {}
        
        for input_config in input_configs:
            self.input_names[input_config['name']] = input_config
            
        for output_config in output_configs:
            self.output_names[output_config['name']] = output_config

        self.confidence_threshold = float(self.model_config.get('parameters', {}).get('confidence_threshold', {'string_value': '0.5'})['string_value'])

    def execute(self, requests):
        responses = []
        
        for request in requests:
            # Get input tensors
            image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
            dets_tensor = pb_utils.get_input_tensor_by_name(request, "dets")
            labels_tensor = pb_utils.get_input_tensor_by_name(request, "labels")
            
            # Convert to numpy arrays
            image = image_tensor.as_numpy()  # Shape: [1, 3, 640, 640]
            dets = dets_tensor.as_numpy()    # Shape: [1, 100, 5] - [x1, y1, x2, y2, score]
            labels = labels_tensor.as_numpy() # Shape: [1, 100] - class labels
            
            # Process each image in the batch
            batch_size = image.shape[0]
            cropped_persons = []
            
            for b in range(batch_size):
                img_data = image[b]  # [3, 640, 640]
                det_data = dets[b]   # [100, 5]
                label_data = labels[b] # [100]
                
                # Convert from CHW to HWC for OpenCV
                img_hwc = np.transpose(img_data, (1, 2, 0))  # [640, 640, 3]
                
                # Convert from float [0,1] to uint8 [0,255] if needed
                if img_hwc.max() <= 1.0:
                    img_hwc = (img_hwc * 255).astype(np.uint8)
                else:
                    img_hwc = img_hwc.astype(np.uint8)
                
                # Find person detections (assuming person class = 0 in COCO)
                person_detections = []
                for i in range(len(det_data)):
                    score = det_data[i, 4]
                    label = label_data[i] if i < len(label_data) else -1
                    
                    # Filter by confidence and person class
                    if score > self.confidence_threshold and label == 0:  # person class
                        x1, y1, x2, y2 = det_data[i, :4]
                        person_detections.append([x1, y1, x2, y2, score])
                
                # If we have person detections, crop the first one (highest confidence)
                if person_detections:
                    # Sort by confidence (highest first)
                    person_detections = sorted(person_detections, key=lambda x: x[4], reverse=True)
                    x1, y1, x2, y2, score = person_detections[0]
                    
                    # Ensure coordinates are within image bounds
                    h, w = img_hwc.shape[:2]
                    x1 = max(0, min(int(x1), w-1))
                    y1 = max(0, min(int(y1), h-1))
                    x2 = max(x1+1, min(int(x2), w))
                    y2 = max(y1+1, min(int(y2), h))
                    
                    # Crop the person region
                    cropped_person = img_hwc[y1:y2, x1:x2]
                    
                    # Resize to standard size for pose estimation (e.g., 256x192)
                    cropped_person = cv2.resize(cropped_person, (192, 256))
                    
                else:
                    # No person detected
                    print("No person detected, returing empty crop")
                    cropped_person = np.zeros((256, 192, 3), dtype=np.uint8)
                
                # Convert back to CHW format and normalize to [0,1]
                cropped_person_chw = np.transpose(cropped_person, (2, 0, 1)).astype(np.float32) / 255.0
                cropped_persons.append(cropped_person_chw)
            
            # Stack all cropped persons
            output_array = np.stack(cropped_persons, axis=0)  # [batch_size, 3, 256, 192]
            
            # Create output tensor
            output_tensor = pb_utils.Tensor("cropped_person", output_array)
            
            # Create inference response
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        pass