Note: A work in progress. Not thoroughly tested. Your mileage may vary.

# Audio Restoration for 78rpm Records

Deep learning-based audio restoration system for enhancing 78rpm vinyl records with denoising and stereo separation capabilities.

**Note:** This project was designed with training on NVIDIA Jetson devices in mind (for experimental edge AI purposes), but works on any CUDA-compatible GPU system or CPU. The model is pretty small. This was an experiment and learning exercise... I wouldn't expect these to produce archival quality audio, but with some fiddling you might be surprised :).

## Features

- **Audio Denoising**: U-Net based model to remove noise, crackles, and pops from vintage recordings
- **Bandwidth Extension**: Super-resolution model to restore high-frequency content and transients (22.05kHz → 44.1kHz)
- **Stereo Separation**: AI-powered mono-to-stereo conversion with spatial enhancement and decorrelation loss
- **Test Generation**: Automatic test output generation during training for quality monitoring
- **End-to-End Pipeline**: Complete workflow from raw audio to restored stereo output
- **Remote Training**: Scripts for training on remote GPU servers (e.g., Jetson devices)

## Project Structure

```
ml-audio-restoration/
├── data/
│   ├── raw/              # Original audio files
│   └── processed/        # Preprocessed audio
├── models/
│   └── checkpoints/      # Saved model checkpoints
├── src/
│   ├── models/           # Model architectures
│   │   ├── denoiser.py
│   │   ├── super_resolution.py
│   │   └── stereo_separator_efficient.py
│   ├── utils/            # Audio processing utilities
│   │   ├── audio_processing.py
│   │   └── preprocessing.py
│   ├── training/         # Training scripts
│   │   ├── trainer.py
│   │   ├── train_denoiser.py
│   │   ├── train_super_resolution.py
│   │   └── train_stereo.py
│   └── inference.py      # Inference script
├── config/               # Configuration files
├── scripts/              # Deployment and remote training scripts
├── notebooks/            # Jupyter notebooks for exploration
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ml-audio-restoration
```

2. Create a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Preparing Your Data

Place your training audio files in `data/raw/`. The system supports WAV, MP3, FLAC, and OGG formats.

For stereo separation training, you'll need stereo audio files. The system will automatically create mono versions for training input.

### Training

**Train the denoiser:**
```bash
python src/training/train_denoiser.py --batch_size 2 --num_epochs 1000 --chunk_duration 2.0
```

**Train the super-resolution model:**
```bash
python src/training/train_super_resolution.py
```

**Train the stereo separator:**
```bash
# Default settings (recommended - matches current checkpoint)
python src/training/train_stereo.py --num_epochs 1000 --batch_size 4 --chunk_duration 2.0

# Or with explicit parameters
python src/training/train_stereo.py --num_epochs 1000 --batch_size 4 --chunk_duration 2.0 --base_channels 32 --lstm_hidden 64
```

**Options:**
- `--num_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 4, reduce if out of memory)
- `--chunk_duration`: Audio chunk duration in seconds (default: 2.0)
- `--base_channels`: Base number of channels in encoder (default: 32)
- `--lstm_hidden`: LSTM hidden size (default: 64)
- `--no_test_gen`: Disable test output generation during training

**Test Output Generation:**
Place test audio files in a directory (e.g., `test_audio/`) and configure `test_audio` in the config. The trainer will automatically generate stereo outputs every 10 epochs for quality monitoring.

Training checkpoints will be saved in `models/checkpoints/`.

### Inference

Restore a 78rpm recording with all enhancements:
```bash
python src/inference.py input_audio.wav output_restored.wav
```

Options:
- `--denoiser`: Path to denoiser checkpoint (default: `models/checkpoints/best_model.pth`)
- `--super-res`: Path to super-resolution checkpoint (default: `models/checkpoints/super_resolution/best_model.pth`)
- `--stereo`: Path to stereo separator checkpoint (default: `models/checkpoints/stereo/best_model.pth`)
- `--sample-rate`: Sample rate for processing (default: 22050)
- `--no-super-res`: Disable bandwidth extension (keep at original sample rate)
- `--device`: Device to use - 'cuda' or 'cpu' (default: auto-detect)

Example with custom checkpoints:
```bash
python src/inference.py input.wav output.wav --denoiser models/custom_denoiser.pth --super-res models/custom_sr.pth
```

Skip bandwidth extension if you want to keep the original frequency range:
```bash
python src/inference.py input.wav output.wav --no-super-res
```

## Models

### AudioDenoiser
- **Architecture**: U-Net with skip connections
- **Input**: Mono audio (1 channel)
- **Output**: Denoised mono audio (1 channel)
- **Purpose**: Remove noise, crackles, pops, and other artifacts

### AudioSuperResolution
- **Architecture**: Residual network with subpixel upsampling
- **Input**: Low-resolution audio (22.05kHz)
- **Output**: High-resolution audio (44.1kHz)
- **Purpose**: Restore high-frequency content and transients lost to degradation
- **Loss Function**: Combined time-domain MSE + spectral loss for frequency accuracy

### StereoSeparator
- **Architecture**: Efficient model with dilated convolutions + LSTM + dual decoders (~500K params)
- **Input**: Mono audio (1 channel)
- **Output**: Stereo audio (2 channels)
- **Purpose**: Create realistic stereo field from mono recordings
- **Loss Function**: MSE reconstruction + stereo decorrelation penalty
      - Helps create more natural stereo separation
  - Optional: Decorrelation loss encourages distinct L/R channels rather than simple duplication

## Training Tips

1. **Data Requirements**: 
   - Minimum 1-2 hours of clean audio for denoiser training
   - Stereo recordings for stereo separation training
   - More diverse data = better generalization

2. **GPU Recommended**: Training on CPU is slow. A GPU with 6GB+ VRAM is recommended.

3. **Hyperparameter Tuning**: Adjust learning rate, batch size, and model size in training scripts based on your data and hardware.

4. **Data Augmentation**: The system automatically simulates vinyl artifacts for denoiser training.

5. **Test Monitoring**: Place a few test audio files in a directory and configure the training script to generate outputs during training. This helps monitor quality improvements across epochs.

6. **Chunk Duration**: For stereo separation, 2-second chunks work well and avoid cuDNN LSTM sequence length limits. Longer chunks may cause memory or compatibility issues.

## Configuration

Edit the config dictionaries in training scripts to adjust:
- Sample rate (default: 22050 Hz)
- Chunk duration (default: 2 seconds)
- Batch size
- Learning rate
- Number of epochs

## Remote Training (Optional)

The `scripts/` directory contains PowerShell and bash scripts for training on remote GPU servers:

- `setup_remote_jetson.ps1`: Sync code and data to remote server
- `sync_data_to_jetson.ps1`: Sync training data only
- `monitor_remote_training.ps1`: Monitor training progress remotely

These are optional and primarily designed for NVIDIA Jetson devices. Local training works fine for most use cases.

## Technical Details

- **Framework**: PyTorch
- **Audio Processing**: torchaudio
- **Default Sample Rate**: 22050 Hz
- **Loss Function**: MSE (Mean Squared Error)
- **Optimizer**: Adam with ReduceLROnPlateau scheduler

## Stereo Separation Approach

The stereo separation uses a learned approach rather than simple panning:
1. Encodes mono audio to feature space
2. LSTM captures temporal dependencies
3. Dual decoders generate distinct left/right channels
4. Spatial processor enhances stereo separation

This creates a more natural stereo field compared to simple delay or frequency-based panning.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchaudio
- numpy
- tqdm

See `requirements.txt` for complete list.

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- U-Net architecture inspired by audio source separation research
- Stereo widening techniques from spatial audio processing literature
