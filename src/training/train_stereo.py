"""
Training script for the stereo separator model with TensorBoard
"""
import sys
sys.path.append('.')

import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from src.models.stereo_separator import StereoSeparator
from src.utils.preprocessing import StereoDataset
from src.training.trainer import Trainer
from torch.utils.data import DataLoader, random_split


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train stereo separator model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--chunk_duration', type=float, default=2.0, help='Audio chunk duration in seconds')
    parser.add_argument('--base_channels', type=int, default=32, help='Base channels for model')
    parser.add_argument('--lstm_hidden', type=int, default=64, help='LSTM hidden size')
    parser.add_argument('--no_test_gen', action='store_true', help='Skip test output generation during training')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data_dir': 'data/raw',  # Should contain stereo audio files
        'sample_rate': 22050,
        'chunk_duration': args.chunk_duration,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'val_split': 0.1,
        'num_workers': 0,  # Set to 0 for Windows to allow Ctrl+C to work
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'log_dir': 'runs/stereo',
        'test_audio': 'test_audio',  # Test audio path (files in root only)
        'test_output': 'outputs/stereo_tests'  # Where to save test outputs
    }
    
    print("=" * 60)
    print("Stereo Separator Training")
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
    full_dataset = StereoDataset(
        data_dir=config['data_dir'],
        sample_rate=config['sample_rate'],
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
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False  # Disable for Jetson to avoid memory issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False  # Disable for Jetson to avoid memory issues
    )
    
    # Create model
    print("\nInitializing model...")
    model = StereoSeparator(
        base_channels=args.base_channels,
        lstm_hidden=args.lstm_hidden,
        num_lstm_layers=1
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
    test_audio_dir = None
    if config['test_audio']:
        from pathlib import Path
        test_path = Path(config['test_audio'])
        if test_path.is_file():
            test_audio_dir = str(test_path.parent)
            print(f"\n✓ Test audio directory: {test_audio_dir}")
        elif test_path.is_dir():
            # Use the directory directly
            test_audio_dir = str(test_path)
            # Count audio files (root only, not subdirectories)
            audio_files = []
            for ext in ['*.wav', '*.flac', '*.mp3', '*.ogg']:
                audio_files.extend(test_path.glob(ext))
            print(f"\n✓ Test audio directory: {test_audio_dir} ({len(audio_files)} files)")
        if not test_audio_dir:
            print(f"\n⚠ Test audio not found: {config['test_audio']}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        device=config['device'],
        checkpoint_dir='models/checkpoints/stereo',
        writer=writer,  # Pass TensorBoard writer
        test_audio_dir=test_audio_dir if not args.no_test_gen else None,  # Test audio directory for checkpoints
        test_output_dir=config['test_output']  # Where to save test outputs
    )
    
    # Check for existing checkpoint to resume from
    from pathlib import Path
    checkpoint_dir = Path('models/checkpoints/stereo')
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
            # Generate test output immediately after loading checkpoint
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
    print(f"Best model saved to models/checkpoints/stereo/best_model.pth")
    print(f"TensorBoard logs: {config['log_dir']}")


if __name__ == "__main__":
    main()
