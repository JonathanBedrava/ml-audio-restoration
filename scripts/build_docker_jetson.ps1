# Build and Run Docker Container on Jetson

$JETSON_USER = "jonathan"
$JETSON_IP = "jetson1"
$LOCAL_PROJECT = "D:\source\repos\ml-audio-restoration"

Write-Host "=== Building Docker Container on Jetson ===" -ForegroundColor Green
Write-Host ""

# Copy Dockerfile to Jetson
Write-Host "Copying Dockerfile..." -ForegroundColor Cyan
scp Dockerfile.jetson ${JETSON_USER}@${JETSON_IP}:/ssd/ml-audio-restoration/Dockerfile.jetson

# Sync project files if needed
Write-Host "Syncing project files..." -ForegroundColor Cyan
.\scripts\setup_remote_jetson.ps1 -CodeOnly

# Build container on Jetson
Write-Host ""
Write-Host "Building Docker image (this takes 5-10 minutes)..." -ForegroundColor Yellow
ssh ${JETSON_USER}@${JETSON_IP} "cd /ssd/ml-audio-restoration && sudo docker build -f Dockerfile.jetson -t ml-audio-restore:latest ."

Write-Host ""
Write-Host "=== Docker Build Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "To run training in Docker:" -ForegroundColor Yellow
Write-Host "  ssh ${JETSON_USER}@${JETSON_IP}" -ForegroundColor White
Write-Host "  cd /ssd/ml-audio-restoration" -ForegroundColor White
Write-Host "  sudo docker run --gpus all -v \`$(pwd)/data:/workspace/ml-audio-restoration/data -v \`$(pwd)/models:/workspace/ml-audio-restoration/models -v \`$(pwd)/outputs:/workspace/ml-audio-restoration/outputs -v \`$(pwd)/runs:/workspace/ml-audio-restoration/runs -it ml-audio-restore:latest" -ForegroundColor White
Write-Host ""
Write-Host "Inside container, run:" -ForegroundColor Yellow
Write-Host "  python src/training/train_denoiser.py" -ForegroundColor White
