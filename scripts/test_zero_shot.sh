#!/bin/bash
# Zero-shot imitation test script for PHC model

# Configuration
MODEL_PATH="output/HumanoidIm/vlm/kick/24422/humanoidIm/Humanoid.pth"
MOTION_FILE="sample_data/loop0/During_form_practice__the_practitioner_performed_basic_elbow_strike_with_short_range_and_moderate_force_with_whirlwind_kick_with_pivot_foot_rotation__tight_core_for_balance_and_target_focused_gaze__fo.pkl"
ACTION_OFFSET="phc/data/action_offset_smpl.pkl"
HUMANOID_TYPE="smpl"
NUM_ENVS=1
NUM_MOTIONS=5

echo "Starting zero-shot imitation test..."
echo "Model: $MODEL_PATH"
echo "Motion: $MOTION_FILE"

# Check if files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$MOTION_FILE" ]; then
    echo "Error: Motion file not found: $MOTION_FILE"
    exit 1
fi

if [ ! -f "$ACTION_OFFSET" ]; then
    echo "Error: Action offset file not found: $ACTION_OFFSET"
    exit 1
fi

# Run the evaluation using the existing eval_in_isaaclab.py
python scripts/eval_in_isaaclab.py \
    --policy_path "$MODEL_PATH" \
    --motion_file "$MOTION_FILE" \
    --action_offset_file "$ACTION_OFFSET" \
    --humanoid_type "$HUMANOID_TYPE" \
    --num_envs "$NUM_ENVS" \
    --num_motions "$NUM_MOTIONS"

echo "Zero-shot imitation test completed!"