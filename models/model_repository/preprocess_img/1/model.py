import numpy as np
import cv2
import triton_python_backend_utils as pb_utils
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class TritonPythonModel:
    """Image preprocessing model for RTMDet detection."""

    def initialize(self, args):
        logger.info("Initializing image preprocessing model...")
        self.model_config = json.loads(args['model_config'])

        # RTMDet preprocessing parameters
        self.input_size = (640, 640)
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                # Get input image bytes
                input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
                img_bytes = input_tensor.as_numpy()[0]

                # Decode and preprocess image
                nparr = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    # log
                    print("No image found, using dummy image")
                    processed = np.zeros((3, 640, 640), dtype=np.float32)
                else:
                    # Resize to RTMDet input size
                    resized = cv2.resize(image, self.input_size)
                    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                    # Normalize
                    normalized = resized.astype(np.float32)
                    normalized = (normalized - self.mean) / self.std
                    processed = normalized.transpose(2, 0, 1)
                    processed = np.expand_dims(processed, axis=0)  # Add batch dimension [1, 3, 640, 640]

                output_tensor = pb_utils.Tensor("PROCESSED_IMAGE", processed.astype(np.float32))
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(response)

            except Exception as e:
                logger.error(f"Error in preprocessing: {e}")
                processed = np.zeros((3, 640, 640), dtype=np.float32)
                output_tensor = pb_utils.Tensor("PROCESSED_IMAGE", processed.astype(np.float32))
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(response)

        return responses

    def finalize(self):
        logger.info("Finalizing preprocessing model.")
