## Feature Normalization Fix - COMPLETED (Sep 13, 2025)

### ✅ **CRITICAL BUG RESOLVED**: Training vs Inference Mismatch

**Problem Identified**: The MLP model was receiving different feature vectors during inference than during training, causing poor performance (weak logits like [-0.04, 0.08] instead of confident predictions).

**Root Cause Analysis**:
1. **Feature Vector Size Mismatch**:
   - **Training** (`data_collector.py`): Generated **51-dimensional** vectors (17 keypoints × 3: x, y, confidence)
   - **Inference** (`feature_normalizer`): Generated **47-dimensional** vectors (15 keypoints × 2 + 17 confidences)

2. **Normalization Logic Differences**:
   - **Training**: Used MMPoseInferencer('human') → simple numpy operations → neck-centered scaling by shoulder distance
   - **Inference**: Used complex MMPose RTMSimCCHead codec → different coordinate system → torso-based scaling

3. **Configuration Mismatches**:
   - `feature_normalizer/config.pbtxt`: Expected 47 dims (WRONG)
   - `mlp_phone_detector/config.pbtxt`: Expected 47 dims (WRONG)
   - Both should expect 51 dims to match training

### 🔧 **Fixes Applied**:

1. **Completely Rewrote `feature_normalizer/1/model.py`**:
   - Removed MMPose RTMSimCCHead dependency to eliminate complexity
   - Implemented simple argmax decoding of simcc_x/simcc_y heatmaps  
   - **Exact replication** of `data_collector.py` normalization logic (lines 71-104)
   - Same confidence threshold (0.3), same neck_proxy calculation, same body_scale logic
   - Outputs exactly 51-dimensional feature vectors

2. **Updated Configuration Files**:
   - `feature_normalizer/config.pbtxt`: Changed output dims from [47] → [51] 
   - `mlp_phone_detector/config.pbtxt`: Changed input dims from [47] → [51]

3. **Maintained Training Compatibility**:
   - Verified `train_mlp.py` uses `INPUT_DIM = 51` ✅
   - Verified `data_collector.py` generates 51-dimensional vectors ✅
   - Pipeline now feeds MLP the same feature format it was trained on ✅

### 🧪 **Ready for Testing**:
The system should now produce confident predictions instead of weak noise. Expected behavior:
- Phone images: Strong positive logits (e.g., `[[-2.5, 3.2]]`)
- No-phone images: Strong negative logits (e.g., `[[1.8, -1.9]]`)

### 🚀 **Next Steps**:
1. **Restart Docker container**: `cd deployment && ./deploy.sh restart`
2. **Run debug pipeline**: `python debug_pipeline.py` 
3. **Verify confident predictions**: Logits should show clear separation instead of noise
4. **Test with multiple images**: Confirm consistent behavior across test set

---

## Plan & Review

### Before starting work
- Write a plan to .claude/tasks/TASK_NAME.md.
- The plan should be a detailed implementation plan and the reasoning behind them, as well as tasks broken down.
- Don't ever over plan, always think MVP.
- Once you write the plan, firstly ask me to review it. Do not continue until I approve the plan.

### While implementing
- You should update the plan as you work.
- After you complete tasks in the plan, you should update and append detailed descriptions of the changes you made, so following tasks can be easily hand over to other engineers.