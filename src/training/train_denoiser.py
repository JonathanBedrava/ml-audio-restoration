"""
Training script for the audio denoiser model with TensorBoard integration
"""
import sys
sys.path.append('.')

import torch
from torch.utils.tensorboard import SummaryWriter
from src.models.denoiser import AudioDenoiser
from src.utils.preprocessing import AudioRestorationDataset
from src.training.trainer import Trainer
from torch.utils.data import DataLoader, random_split


def main():
    # Configuration
    config = {
        'data_dir': 'data/raw',
        'sample_rate': 22050,
        'chunk_duration': 1.0,  # Reduced to 1 second for Jetson memory constraints (676K param U-Net)
        'batch_size': 1,  # Minimum batch size for Jetson memory constraints
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'val_split': 0.1,
        'num_workers': 0,  # Set to 0 for Windows to allow Ctrl+C to work
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'log_dir': 'runs/denoiser',
        'test_audio': 'test_audio',  # Test audio directory
        'test_output': 'outputs/denoiser_tests'
    }
    
    print("=" * 60)
    print("Audio Denoiser Training")
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
    full_dataset = AudioRestorationDataset(
        data_dir=config['data_dir'],
        sample_rate=config['sample_rate'],
        chunk_duration=config['chunk_duration'],
        add_artifacts=True
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
    model = AudioDenoiser(in_channels=1, out_channels=1)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create TensorBoard writer
    writer = SummaryWriter(config['log_dir'])
    writer.add_text('Config', str(config))
    writer.add_text('Model/Parameters', f'Total: {total_params:,}, Trainable: {trainable_params:,}')
    
    # Check if test audio directory exists
    test_audio_dir = None
    if config['test_audio']:
        from pathlib import Path
        test_path = Path(config['test_audio'])
        if test_path.is_dir():
            # Count test files in root directory only
            audio_files = []
            for ext in ['*.wav', '*.flac', '*.mp3', '*.ogg']:
                audio_files.extend(test_path.glob(ext))
            
            if audio_files:
                test_audio_dir = str(test_path)
                print(f"\n✓ Test audio directory: {test_audio_dir} ({len(audio_files)} files)")
            else:
                print(f"\n⚠ No audio files found in: {config['test_audio']}")
        elif test_path.is_file():
            # Single file - use parent directory
            test_audio_dir = str(test_path.parent)
            print(f"\n✓ Test audio directory: {test_audio_dir}")
        else:
            print(f"\n⚠ Test audio not found: {config['test_audio']}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        device=config['device'],
        checkpoint_dir='models/checkpoints/denoiser',
        writer=writer,
        test_audio_dir=test_audio_dir,  # Use directory instead of single file
        test_output_dir=config['test_output']
    )
    
    # Check for existing checkpoint to resume from
    from pathlib import Path
    checkpoint_dir = Path('models/checkpoints/denoiser')
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
            
            # Generate test output immediately after resuming
            if test_audio_dir:
                print("\nGenerating test outputs from resumed checkpoint...")
                trainer.generate_test_output(f'resumed_epoch_{trainer.epoch}')
        else:
            print("\nNo checkpoints found, starting fresh training...")
    else:
        print("\nNo checkpoints found, starting fresh training...")
    
    # Train
    print("\nStarting training...")
    print(f"TensorBoard: tensorboard --logdir={config['log_dir']}")
    if test_audio_dir:
        print(f"Test outputs will be saved to: {config['test_output']}")
    trainer.train(num_epochs=config['num_epochs'], save_every=10)
    
    writer.close()
    print(f"\n✓ Training complete!")
    print(f"Best model saved to models/checkpoints/denoiser/best_model.pth")
    print(f"TensorBoard logs: {config['log_dir']}")


if __name__ == "__main__":
    main()
