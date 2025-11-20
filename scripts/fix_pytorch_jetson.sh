#!/bin/bash
# Fix PyTorch installation for Jetson
# PyTorch from pip doesn't work on Jetson - need NVIDIA's builds

set -e

echo "=== Fixing PyTorch for Jetson ==="
echo ""

# Initialize conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

# Activate conda environment
conda activate audio-restore

# Check JetPack version
echo "Checking system info..."
echo "CUDA version:"
nvcc --version || echo "nvcc not found"
echo ""
echo "Python version:"
python --version

# Uninstall existing PyTorch
echo ""
echo "Removing incompatible PyTorch..."
pip uninstall -y torch torchvision torchaudio || true

# Remove the symlink we created
echo "Cleaning up symlinks..."
sudo rm -f /usr/lib/aarch64-linux-gnu/libcudnn.so.8

# Install PyTorch built for JetPack 6.2 with cuDNN 9
echo ""
echo "Installing NVIDIA PyTorch for JetPack 6.x..."
echo "This may take a few minutes..."

pip3 install --upgrade pip
pip3 install numpy==1.26.4

# PyTorch 2.5.0 for JetPack 6.x (should work with 6.2)
echo "Downloading PyTorch 2.5.0 for JetPack 6.x..."
# Try v60 path which should work for all JetPack 6.x
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl -O /tmp/torch.whl

if [ $? -ne 0 ]; then
    echo "Direct download failed, trying alternate source..."
    # Fallback to pip with NVIDIA index
    pip3 install torch --index-url https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch || \
    pip3 install torch  # Last resort: standard PyTorch
else
    pip3 install /tmp/torch.whl
    rm /tmp/torch.whl
fi

# Install torchaudio
pip3 install torchaudio

echo ""
echo "Verifying installation..."
if python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"; then
    echo ""
    echo "=== PyTorch fix complete! ==="
    echo ""
    echo "Ready to train!"
else
    echo ""
    echo "ERROR: PyTorch still not working!"
    echo ""
    echo "cuDNN 8/9 compatibility issue. This may require building PyTorch from source."
    exit 1
fi
