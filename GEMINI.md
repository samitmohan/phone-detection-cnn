# Phone Detection CNN Debugging Plan

## Problem

The `mlp_phone_detector` model is producing incorrect results. The root cause is a mismatch between the keypoint normalization logic used during training (`data_collector.py`) and the logic used during inference in the `feature_normalizer` Triton model.

## Analysis

- The overall pipeline structure is correct.
- The `MMPoseInferencer('human', ...)` used in training correctly corresponds to the `rtmdet_detection` and `rtmpose_estimation` models in the Triton pipeline.
- The `feature_normalizer` model is not correctly replicating the normalization logic from the training script, causing the `mlp_phone_detector` to receive incorrectly formatted feature vectors.

## Plan

1.  **Correct the `feature_normalizer` model:** The `feature_normalizer/1/model.py` script will be updated to perfectly match the normalization logic from `training/data_collector.py`.
2.  **Verify the fix:** The `debug_pipeline.py` script will be used to run the entire pipeline and inspect the `normalized_features` and `mlp_logits` to confirm that the fix is working correctly.
