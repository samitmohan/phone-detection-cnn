import json
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    """
    Postprocessing model for phone detection pipeline.
    
    This model:
    1. Receives raw logits from the MLP model
    2. Applies softmax to get probabilities
    3. Applies confidence threshold (0.5) for classification
    4. Formats results with person IDs and confidence scores
    5. Returns structured prediction results
    """

    def initialize(self, args):
        """Initialize the postprocessing model."""
        
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Get input/output configuration
        input_config = pb_utils.get_input_config_by_name(
            model_config, "MODEL_OUTPUT")
        self.input_dtype = pb_utils.triton_string_to_numpy(
            input_config['data_type'])
        
        # Outputs
        person_id_config = pb_utils.get_output_config_by_name(
            model_config, "PERSON_ID")
        self.person_id_dtype = pb_utils.triton_string_to_numpy(
            person_id_config['data_type'])
        
        is_phone_config = pb_utils.get_output_config_by_name(
            model_config, "IS_PHONE")
        self.is_phone_dtype = pb_utils.triton_string_to_numpy(
            is_phone_config['data_type'])
        
        confidence_config = pb_utils.get_output_config_by_name(
            model_config, "CONFIDENCE")
        self.confidence_dtype = pb_utils.triton_string_to_numpy(
            confidence_config['data_type'])
        
        # Classification threshold
        self.confidence_threshold = 0.5
        
        print("Postprocessing model initialized")

    def _softmax(self, x):
        """Apply softmax function to convert logits to probabilities."""
        # Numerical stability: subtract max value
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _process_predictions(self, model_outputs):
        """
        Process model outputs to extract predictions.
        
        Args:
            model_outputs: Model logits of shape [batch_size, 2]
            
        Returns:
            tuple: (person_ids, is_phone_flags, confidence_scores)
        """
        # Apply softmax to get probabilities
        probabilities = self._softmax(model_outputs)
        
        # Extract phone class probabilities (class 1)
        phone_probabilities = probabilities[:, 1]
        
        # Apply confidence threshold
        is_phone_predictions = (phone_probabilities > self.confidence_threshold).astype(bool)
        
        # Create person IDs (sequential numbering)
        person_ids = np.arange(len(model_outputs), dtype=np.int32)
        
        return person_ids, is_phone_predictions, phone_probabilities.astype(np.float32)

    def execute(self, requests):
        """Execute postprocessing for a batch of requests."""
        responses = []
        
        for request in requests:
            try:
                # Get model output logits
                model_output = pb_utils.get_input_tensor_by_name(request, "MODEL_OUTPUT")
                logits = model_output.as_numpy()
                
                # Process predictions
                person_ids, is_phone_flags, confidence_scores = self._process_predictions(logits)
                
                # Create output tensors
                person_id_tensor = pb_utils.Tensor("PERSON_ID", 
                    person_ids.astype(self.person_id_dtype))
                
                is_phone_tensor = pb_utils.Tensor("IS_PHONE", 
                    is_phone_flags.astype(self.is_phone_dtype))
                
                confidence_tensor = pb_utils.Tensor("CONFIDENCE", 
                    confidence_scores.astype(self.confidence_dtype))
                
                # Create response
                response = pb_utils.InferenceResponse(
                    output_tensors=[person_id_tensor, is_phone_tensor, confidence_tensor]
                )
                responses.append(response)
                
            except Exception as e:
                print(f"Error in postprocessing: {e}")
                
                # Create error response with default values
                batch_size = 1  # Default fallback
                try:
                    if 'model_output' in locals():
                        batch_size = logits.shape[0] if logits.ndim > 0 else 1
                except:
                    pass
                
                # Default outputs for error case
                person_ids = np.arange(batch_size, dtype=self.person_id_dtype)
                is_phone_flags = np.zeros(batch_size, dtype=self.is_phone_dtype)
                confidence_scores = np.zeros(batch_size, dtype=self.confidence_dtype)
                
                person_id_tensor = pb_utils.Tensor("PERSON_ID", person_ids)
                is_phone_tensor = pb_utils.Tensor("IS_PHONE", is_phone_flags)
                confidence_tensor = pb_utils.Tensor("CONFIDENCE", confidence_scores)
                
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[person_id_tensor, is_phone_tensor, confidence_tensor],
                    error=pb_utils.TritonError(f"Postprocessing error: {str(e)}")
                )
                responses.append(error_response)
        
        return responses

    def finalize(self):
        """Clean up resources."""
        print('Postprocessing model done')