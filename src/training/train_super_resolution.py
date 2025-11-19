"""
Training script for the audio super-resolution model with TensorBoard
"""
import sys
sys.path.append('.')

import torch
from torch.utils.tensorboard import SummaryWriter
from src.models.super_resolution import AudioSuperResolution, SpectralLoss
from src.utils.preprocessing import AudioRestorationDataset
from src.training.trainer import Trainer
from torch.utils.data import DataLoader, Dataset, random_split
from src.utils.audio_processing import load_audio, normalize_audio
from pathlib import Path
import numpy as np


class SuperResolutionDataset(Dataset):
    """
    Dataset for audio super-resolution training.
    Pairs low-resolution (22.05kHz) with high-resolution (44.1kHz) audio.
    """
    
    def __init__(
        self,
        data_dir: str,
        low_sample_rate: int = 22050,
        high_sample_rate: int = 44100,
        chunk_duration: float = 2.0
    ):
        self.data_dir = Path(data_dir)
        self.low_sr = low_sample_rate
        self.high_sr = high_sample_rate
        self.chunk_size_low = int(low_sample_rate * chunk_duration)
        self.chunk_size_high = int(high_sample_rate * chunk_duration)
        
        # Find all audio files
        self.audio_files = []
        for ext in ['*.wav', '*.flac']:
            self.audio_files.extend(self.data_dir.glob(f"**/{ext}"))
        
        print(f"Found {len(self.audio_files)} audio files for super-resolution training")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Load at high sample rate
        high_res, _ = load_audio(str(audio_path), self.high_sr, mono=True)
        high_res = normalize_audio(high_res)
        
        # Ensure consistent chunk size
        if high_res.shape[-1] < self.chunk_size_high:
            padding = self.chunk_size_high - high_res.shape[-1]
            high_res = torch.nn.functional.pad(high_res, (0, padding))
        else:
            start = np.random.randint(0, high_res.shape[-1] - self.chunk_size_high + 1)
            high_res = high_res[..., start:start + self.chunk_size_high]
        
        # Create low-resolution version by downsampling
        low_res = torch.nn.functional.interpolate(
            high_res.unsqueeze(0),
            size=self.chunk_size_low,
            mode='linear',
            align_corners=False
        ).squeeze(0)
        
        return low_res, high_res


def main():
    # Configuration
    config = {
        'data_dir': 'data/raw',
        'low_sample_rate': 22050,
        'high_sample_rate': 44100,
        'chunk_duration': 2.0,
        'batch_size': 24,  # Increased for RTX 4090
        'num_epochs': 100,  # Full training
        'learning_rate': 1e-4,
        'val_split': 0.1,
        'num_workers': 0,  # Set to 0 for Windows to allow Ctrl+C to work
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'upscale_factor': 2,  # 2x = 22.05kHz -> 44.1kHz
        'log_dir': 'runs/super_resolution',
        'test_audio': 'G:/raw/opera',  # Test audio path
        'test_output': 'outputs/super_resolution_tests'  # Where to save test outputs
    }
    
    print("=" * 60)
    print("Audio Super-Resolution Training")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create dataset
    print("\nLoading dataset...")
    full_dataset = SuperResolutionDataset(
        data_dir=config['data_dir'],
        low_sample_rate=config['low_sample_rate'],
        high_sample_rate=config['high_sample_rate'],
        chunk_duration=config['chunk_duration']
    )
    
    # Split into train and validation
    val_size = int(len(full_dataset) * config['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    print("\nInitializing model...")
    model = AudioSuperResolution(
        upscale_factor=config['upscale_factor'],
        channels=1,
        base_channels=64,
        num_residual_blocks=8
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create TensorBoard writer
    writer = SummaryWriter(config['log_dir'])
    writer.add_text('Config', str(config))
    writer.add_text('Model/Parameters', f'Total: {total_params:,}, Trainable: {trainable_params:,}')
    
    # Check if test audio exists
    test_audio_path = None
    if config['test_audio']:
        from pathlib import Path
        test_path = Path(config['test_audio'])
        if test_path.is_file():
            test_audio_path = str(test_path)
            print(f"\n✓ Test audio: {test_audio_path}")
        elif test_path.is_dir():
            # Look for a descriptive opera file - prefer ones with known composers
            preferred_files = ['Ave Maria', 'Amore ti vieta', 'Nessun dorma']
            files = []
            for ext in ['*.wav', '*.flac', '*.mp3']:
                files.extend(test_path.glob(ext))
            
            # Try to find a preferred file
            for pref in preferred_files:
                for f in files:
                    if pref.lower() in f.name.lower():
                        test_audio_path = str(f)
                        break
                if test_audio_path:
                    break
            
            # If no preferred file found, use first file
            if not test_audio_path and files:
                test_audio_path = str(files[0])
            
            if test_audio_path:
                print(f"\n✓ Test audio: {test_audio_path}")
        if not test_audio_path:
            print(f"\n⚠ Test audio not found: {config['test_audio']}")
    
    # Create trainer with spectral loss
    print("\nCreating trainer with spectral loss...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        device=config['device'],
        checkpoint_dir='models/checkpoints/super_resolution',
        writer=writer,  # Pass TensorBoard writer
        test_audio_path=test_audio_path,  # Test audio for checkpoints
        test_output_dir=config['test_output']  # Where to save test outputs
    )
    
    # Replace MSE loss with spectral loss
    trainer.criterion = SpectralLoss(n_fft=2048, hop_length=512, alpha=0.5)
    
    # Check for existing checkpoint to resume from
    from pathlib import Path
    checkpoint_dir = Path('models/checkpoints/super_resolution')
    checkpoint_to_load = None
    
    if checkpoint_dir.exists():
        # First check for numbered epoch checkpoints
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if checkpoints:
            checkpoint_to_load = checkpoints[-1]
        # If no epoch checkpoints, check for best_model
        elif (checkpoint_dir / 'best_model.pth').exists():
            checkpoint_to_load = checkpoint_dir / 'best_model.pth'
        
        if checkpoint_to_load:
            print(f"\n✓ Found checkpoint: {checkpoint_to_load}")
            print("Resuming training from checkpoint...")
            trainer.load_checkpoint(checkpoint_to_load.name)
        else:
            print("\nNo checkpoints found, starting fresh training...")
    else:
        print("\nNo checkpoints found, starting fresh training...")
    
    # Train
    print("\nStarting training...")
    print(f"TensorBoard: tensorboard --logdir={config['log_dir']}")
    if test_audio_path:
        print(f"Test outputs will be saved to: {config['test_output']}")
    trainer.train(num_epochs=config['num_epochs'], save_every=10)
    
    writer.close()
    print(f"\n✓ Training complete!")
    print(f"Best model saved to models/checkpoints/super_resolution/best_model.pth")
    print(f"TensorBoard logs: {config['log_dir']}")


if __name__ == "__main__":
    main()
