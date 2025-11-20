# Initial Setup for Jetson Remote Training
# Run this ONCE from Windows to set everything up
# Usage: 
#   .\scripts\setup_remote_jetson.ps1           - Full setup
#   .\scripts\setup_remote_jetson.ps1 -CodeOnly - Sync only code files

param(
    [switch]$CodeOnly
)

$JETSON_USER = "jonathan"
$JETSON_IP = "jetson1"
$LOCAL_PROJECT = "D:\source\repos\ml-audio-restoration"
$REMOTE_PROJECT = "/ssd/ml-audio-restoration"

if ($CodeOnly) {
    Write-Host "=== Syncing Code to Jetson ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "Syncing source code, scripts, and config..." -ForegroundColor Cyan
    
    # Clean up local pycache before syncing
    Write-Host "Cleaning Python cache files..." -ForegroundColor Gray
    Get-ChildItem -Path "$LOCAL_PROJECT\src" -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path "$LOCAL_PROJECT\src" -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
    
    # Use rsync-style single connection or batch scp
    Push-Location $LOCAL_PROJECT
    
    # Single scp command for all directories and files
    scp -r src scripts config environment.yml requirements.txt Dockerfile.jetson ${JETSON_USER}@${JETSON_IP}:${REMOTE_PROJECT}/
    
    # Make scripts executable
    ssh ${JETSON_USER}@${JETSON_IP} "chmod +x ${REMOTE_PROJECT}/scripts/*.sh"
    
    Pop-Location
    
    Write-Host ""
    Write-Host "=== Code sync complete! ===" -ForegroundColor Green
    exit 0
}

Write-Host "=== ML Audio Restoration - Initial Jetson Setup ===" -ForegroundColor Green
Write-Host ""

# Step 1: Create directory on Jetson with proper permissions
Write-Host "Step 1: Creating project directory on Jetson SSD..." -ForegroundColor Cyan
Write-Host "You may be prompted for your sudo password on the Jetson..." -ForegroundColor Yellow
ssh -t ${JETSON_USER}@${JETSON_IP} "sudo mkdir -p ${REMOTE_PROJECT} && sudo chown -R ${JETSON_USER}:${JETSON_USER} ${REMOTE_PROJECT}"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create directory on Jetson" -ForegroundColor Red
    exit 1
}

# Step 2: Sync complete project structure using tar over SSH
Write-Host ""
Write-Host "Step 2: Syncing project files to Jetson..." -ForegroundColor Cyan
Write-Host "This will take a few minutes..." -ForegroundColor Yellow

# Create temporary tar file
Push-Location $LOCAL_PROJECT
$tempTar = "$env:TEMP\ml-audio-setup.tar.gz"
Write-Host "Creating archive..." -ForegroundColor Gray
tar -czf $tempTar --exclude='data/raw' --exclude='data/processed' --exclude='models/checkpoints' --exclude='outputs' --exclude='runs' --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' .

# Copy to Jetson and extract
Write-Host "Transferring to Jetson..." -ForegroundColor Gray
scp $tempTar ${JETSON_USER}@${JETSON_IP}:/tmp/ml-audio-setup.tar.gz
ssh ${JETSON_USER}@${JETSON_IP} "cd ${REMOTE_PROJECT} && tar -xzf /tmp/ml-audio-setup.tar.gz && rm /tmp/ml-audio-setup.tar.gz"

# Cleanup
Remove-Item $tempTar -ErrorAction SilentlyContinue
Pop-Location

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to sync files to Jetson" -ForegroundColor Red
    exit 1
}

# Step 3: Install Miniconda if needed
Write-Host ""
Write-Host "Step 3: Checking for conda on Jetson..." -ForegroundColor Cyan

$condaExists = ssh ${JETSON_USER}@${JETSON_IP} "command -v conda"
if (-not $condaExists) {
    Write-Host "Conda not found. Installing Miniconda..." -ForegroundColor Yellow
    ssh ${JETSON_USER}@${JETSON_IP} @"
cd /tmp
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh -b -p \$HOME/miniconda3
\$HOME/miniconda3/bin/conda init bash
rm Miniconda3-latest-Linux-aarch64.sh
"@
    Write-Host "Conda installed. Shell needs to be reloaded." -ForegroundColor Green
}

# Step 4: Create conda environment
Write-Host ""
Write-Host "Step 4: Setting up conda environment..." -ForegroundColor Cyan
ssh ${JETSON_USER}@${JETSON_IP} @"
source ~/.bashrc
cd ${REMOTE_PROJECT}
if conda env list | grep -q audio-restore; then
    echo 'Environment exists, skipping...'
else
    echo 'Creating environment (this takes 10-15 minutes)...'
    conda env create -f environment.yml
    
    # Install PyTorch with CUDA support via pip (better for Jetson)
    source ~/.bashrc
    conda activate audio-restore
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    echo 'Verifying CUDA...'
    python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'
fi
"@

# Step 5: Sync training data
Write-Host ""
Write-Host "Step 5: Syncing training data..." -ForegroundColor Cyan
Write-Host "This will take a while (310 audio files)..." -ForegroundColor Yellow

Push-Location "${LOCAL_PROJECT}\data\raw"
$tempDataTar = "$env:TEMP\ml-audio-data.tar.gz"
Write-Host "Creating data archive..." -ForegroundColor Gray
tar -czf $tempDataTar .

Write-Host "Transferring data to Jetson..." -ForegroundColor Gray
scp $tempDataTar ${JETSON_USER}@${JETSON_IP}:/tmp/ml-audio-data.tar.gz
ssh ${JETSON_USER}@${JETSON_IP} "mkdir -p ${REMOTE_PROJECT}/data/raw && cd ${REMOTE_PROJECT}/data/raw && tar -xzf /tmp/ml-audio-data.tar.gz && rm /tmp/ml-audio-data.tar.gz"

Remove-Item $tempDataTar -ErrorAction SilentlyContinue
Pop-Location

# Step 6: Sync test audio
Write-Host ""
Write-Host "Step 6: Syncing test audio..." -ForegroundColor Cyan

Push-Location "G:\raw\opera"
$tempTestTar = "$env:TEMP\ml-audio-test.tar.gz"
Write-Host "Creating test audio archive..." -ForegroundColor Gray
tar -czf $tempTestTar .

Write-Host "Transferring test audio to Jetson..." -ForegroundColor Gray
scp $tempTestTar ${JETSON_USER}@${JETSON_IP}:/tmp/ml-audio-test.tar.gz
ssh ${JETSON_USER}@${JETSON_IP} "mkdir -p ${REMOTE_PROJECT}/test_audio/opera && cd ${REMOTE_PROJECT}/test_audio/opera && tar -xzf /tmp/ml-audio-test.tar.gz && rm /tmp/ml-audio-test.tar.gz"

Remove-Item $tempTestTar -ErrorAction SilentlyContinue
Pop-Location

# Step 7: Create config file
Write-Host ""
Write-Host "Step 7: Creating configuration..." -ForegroundColor Cyan

$windowsIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.InterfaceAlias -notmatch 'Loopback'} | Select-Object -First 1).IPAddress

ssh ${JETSON_USER}@${JETSON_IP} @"
cat > ~/.ml_audio_restore_config << 'EOF'
WINDOWS_IP=${windowsIP}
WINDOWS_USER=${env:USERNAME}
WINDOWS_PROJECT_PATH=/d/source/repos/ml-audio-restoration
EOF
"@

Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Project location on Jetson: ${REMOTE_PROJECT}" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start training on Jetson:" -ForegroundColor Yellow
Write-Host "  ssh ${JETSON_USER}@${JETSON_IP}" -ForegroundColor White
Write-Host "  cd ${REMOTE_PROJECT}" -ForegroundColor White
Write-Host "  conda activate audio-restore" -ForegroundColor White
Write-Host "  ./scripts/train_remote.sh denoiser" -ForegroundColor White
Write-Host ""
Write-Host "To monitor from Windows:" -ForegroundColor Yellow
Write-Host "  .\scripts\monitor_remote_training.ps1" -ForegroundColor White
Write-Host ""
