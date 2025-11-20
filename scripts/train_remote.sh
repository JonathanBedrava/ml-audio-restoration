#!/bin/bash
# Remote training script for Jetson with auto-sync back to Windows PC

set -e

# Load Windows PC config
if [ ! -f ~/.ml_audio_restore_config ]; then
    echo "Error: Config file not found. Run setup_jetson.sh first."
    exit 1
fi

source ~/.ml_audio_restore_config

# Get model type from argument
MODEL=${1:-denoiser}
if [[ ! "$MODEL" =~ ^(denoiser|super_resolution|stereo)$ ]]; then
    echo "Usage: $0 [denoiser|super_resolution|stereo]"
    exit 1
fi

echo "=== Training $MODEL model on Jetson ==="
echo "Syncing results to: ${WINDOWS_USER}@${WINDOWS_IP}:${WINDOWS_PROJECT_PATH}"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate audio-restore

# Update test audio path for local Jetson paths
# Point to the synced opera directory
sed -i "s|'test_audio': 'G:/raw/opera'|'test_audio': '/ssd/ml-audio-restoration/test_audio/opera'|g" \
    src/training/train_${MODEL}.py

echo "Test audio path updated to: /ssd/ml-audio-restoration/test_audio/opera"
echo ""

# Start background sync process
(
    while true; do
        sleep 300  # Sync every 5 minutes
        
        echo "[$(date)] Syncing checkpoints and outputs..."
        
        # Sync model checkpoints
        rsync -avz --progress models/checkpoints/ \
            ${WINDOWS_USER}@${WINDOWS_IP}:${WINDOWS_PROJECT_PATH}/models/checkpoints/ 2>/dev/null || true
        
        # Sync test outputs
        rsync -avz --progress outputs/ \
            ${WINDOWS_USER}@${WINDOWS_IP}:${WINDOWS_PROJECT_PATH}/outputs/ 2>/dev/null || true
        
        # Sync TensorBoard logs
        rsync -avz --progress runs/ \
            ${WINDOWS_USER}@${WINDOWS_IP}:${WINDOWS_PROJECT_PATH}/runs/ 2>/dev/null || true
        
        echo "[$(date)] Sync complete"
    done
) &
SYNC_PID=$!

# Trap to kill sync process on exit
trap "kill $SYNC_PID 2>/dev/null; echo 'Final sync...'; \
    rsync -avz models/checkpoints/ ${WINDOWS_USER}@${WINDOWS_IP}:${WINDOWS_PROJECT_PATH}/models/checkpoints/; \
    rsync -avz outputs/ ${WINDOWS_USER}@${WINDOWS_IP}:${WINDOWS_PROJECT_PATH}/outputs/; \
    rsync -avz runs/ ${WINDOWS_USER}@${WINDOWS_IP}:${WINDOWS_PROJECT_PATH}/runs/; \
    echo 'Training stopped and final sync complete'" EXIT

# Start training
echo "Starting training... (Ctrl+C to stop)"
echo "Checkpoints will sync to Windows PC every 5 minutes"
echo ""

python src/training/train_${MODEL}.py

# Final sync happens in trap
wait
