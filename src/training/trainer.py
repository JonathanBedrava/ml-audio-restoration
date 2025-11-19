import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict
import time
import signal
import sys
from tqdm import tqdm


class Trainer:
    """
    Trainer class for audio restoration models with TensorBoard integration
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'models/checkpoints',
        writer: Optional[SummaryWriter] = None,
        test_audio_path: Optional[str] = None,
        test_output_dir: Optional[str] = None
    ):
        """
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save model checkpoints
            writer: TensorBoard SummaryWriter (optional)
            test_audio_path: Path to test audio file for checkpoint testing (optional)
            test_output_dir: Directory to save test outputs (optional)
        """
        # Require GPU for training
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required for training. CUDA not available.")
        
        device = 'cuda'
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = writer
        self.test_audio_path = test_audio_path
        self.test_output_dir = Path(test_output_dir) if test_output_dir else None
        
        if self.test_output_dir:
            self.test_output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = writer
        
        # Loss function - MSE for audio reconstruction
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Mixed precision training for RTX 4090
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.epoch = 0
        self.global_step = 0
        self.interrupted = False
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n⚠️  Training interrupted! Exiting...")
        self.interrupted = True
        # Force exit immediately
        import os
        os._exit(0)
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for batch_idx, (degraded, clean) in enumerate(pbar):
            # Check for interruption
            if self.interrupted:
                break
            
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                output = self.model(degraded)
                loss = self.criterion(output, clean)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Log to TensorBoard every 50 batches
            if self.writer and batch_idx % 50 == 0:
                self.writer.add_scalar('Loss/train_batch', loss.item(), self.global_step)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
            pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model"""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for degraded, clean in tqdm(self.val_loader, desc='Validation'):
                degraded = degraded.to(self.device)
                clean = clean.to(self.device)
                
                output = self.model(degraded)
                loss = self.criterion(output, clean)
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self, num_epochs: int, save_every: int = 5):
        """
        Train the model for specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Start from current epoch if resuming
        start_epoch = self.epoch
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch}")
        
        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch + 1
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Check for interruption
            if self.interrupted:
                # Already handled in signal handler
                break
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            if self.val_loader is not None:
                self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Log to TensorBoard
            epoch_time = time.time() - start_time
            if self.writer:
                self.writer.add_scalar('Loss/train_epoch', train_loss, self.epoch)
                if self.val_loader is not None:
                    self.writer.add_scalar('Loss/val_epoch', val_loss, self.epoch)
                self.writer.add_scalar('Time/epoch_duration', epoch_time, self.epoch)
                
                # Log audio samples every 10 epochs
                if self.epoch % 10 == 0 and self.val_loader is not None:
                    self.log_audio_samples()
            
            # Print epoch summary
            print(f"\nEpoch {self.epoch}/{num_epochs} - {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.6f}")
            if self.val_loader is not None:
                print(f"Val Loss: {val_loss:.6f}")
            
            # Save checkpoint
            if self.epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.epoch}.pth')
                # Generate test output if test audio provided
                if self.test_audio_path:
                    self.generate_test_output(f'epoch_{self.epoch}')
            
            # Save best model
            if self.val_loader is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                print(f"✓ Saved best model with val_loss: {val_loss:.6f}")
                # Generate test output for best model
                if self.test_audio_path:
                    self.generate_test_output('best')
        
        print("✓ Training complete!")
    
    def log_audio_samples(self):
        """Log audio samples to TensorBoard"""
        if self.writer is None or self.val_loader is None:
            return
        
        self.model.eval()
        with torch.no_grad():
            # Get one batch
            degraded, clean = next(iter(self.val_loader))
            degraded = degraded[:1].to(self.device)
            clean = clean[:1].to(self.device)
            
            # Generate output
            with torch.amp.autocast('cuda'):
                output = self.model(degraded)
            
            # Log to TensorBoard (assuming 22050 Hz sample rate)
            self.writer.add_audio('Audio/degraded', degraded[0].cpu(), self.epoch, sample_rate=22050)
            self.writer.add_audio('Audio/clean', clean[0].cpu(), self.epoch, sample_rate=22050)
            self.writer.add_audio('Audio/restored', output[0].cpu(), self.epoch, sample_rate=22050)
    
    def generate_test_output(self, suffix: str):
        """Generate denoised output from test audio file (first 30 seconds)"""
        if not self.test_audio_path or not self.test_output_dir:
            return
        
        import torchaudio
        import soundfile as sf
        
        print(f"  Generating test output: {suffix}")
        
        self.model.eval()
        with torch.no_grad():
            # Load test audio
            waveform, sample_rate = torchaudio.load(self.test_audio_path)
            
            # Save original audio (before any processing)
            original_path = self.test_output_dir / f'original_{suffix}.wav'
            if waveform.shape[0] > 1:
                # Stereo - convert to mono for saving
                original_mono = waveform.mean(dim=0, keepdim=True)
            else:
                original_mono = waveform
            # Take first 30 seconds of original
            max_samples_orig = sample_rate * 30
            if original_mono.shape[1] > max_samples_orig:
                original_mono = original_mono[:, :max_samples_orig]
            sf.write(
                str(original_path),
                original_mono.squeeze(0).numpy(),
                sample_rate
            )
            
            # Convert to mono if needed for processing
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample to 22050 if needed (for denoiser)
            if sample_rate != 22050:
                resampler = torchaudio.transforms.Resample(sample_rate, 22050)
                waveform = resampler(waveform)
                sample_rate = 22050
            
            # Take only first 30 seconds
            max_samples = 22050 * 30  # 30 seconds at 22050 Hz
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            
            # Move to device
            waveform = waveform.to(self.device)
            
            # Process in chunks to avoid OOM
            chunk_size = 22050 * 10  # 10 second chunks
            output_chunks = []
            
            for i in range(0, waveform.shape[1], chunk_size):
                chunk = waveform[:, i:i+chunk_size]
                
                # Pad if necessary
                if chunk.shape[1] < chunk_size:
                    padding = chunk_size - chunk.shape[1]
                    chunk = torch.nn.functional.pad(chunk, (0, padding))
                    had_padding = True
                else:
                    had_padding = False
                
                # Add batch dimension
                chunk = chunk.unsqueeze(0)
                
                # Process
                with torch.amp.autocast('cuda'):
                    restored_chunk = self.model(chunk)
                
                # Remove batch dimension and padding
                restored_chunk = restored_chunk.squeeze(0)
                if had_padding:
                    restored_chunk = restored_chunk[:, :-padding]
                
                output_chunks.append(restored_chunk.cpu())
            
            # Concatenate all chunks
            restored = torch.cat(output_chunks, dim=1)
            
            # Convert to float32 for saving
            restored = restored.float()
            waveform_cpu = waveform.cpu().float()
            
            # Save degraded audio
            degraded_path = self.test_output_dir / f'degraded_{suffix}.wav'
            sf.write(
                str(degraded_path),
                waveform_cpu.squeeze(0).numpy(),
                sample_rate
            )
            
            # Save restored audio
            output_path = self.test_output_dir / f'denoised_{suffix}.wav'
            sf.write(
                str(output_path),
                restored.squeeze(0).numpy(),
                sample_rate
            )
            
            # Save comparison (degraded left channel, restored right channel for stereo comparison)
            comparison = torch.stack([waveform_cpu.squeeze(0), restored.squeeze(0)], dim=0)
            comparison_path = self.test_output_dir / f'comparison_{suffix}.wav'
            sf.write(
                str(comparison_path),
                comparison.numpy().T,  # Transpose for soundfile (samples, channels)
                sample_rate
            )
            print(f"  ✓ Test outputs saved: {output_path.parent} (30s samples)")
            print(f"    - original_{suffix}.wav (clean source audio)")
            print(f"    - degraded_{suffix}.wav (with simulated artifacts)")
            print(f"    - denoised_{suffix}.wav (model output)")
            print(f"    - comparison_{suffix}.wav (stereo: L=degraded, R=denoised)")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}")


if __name__ == "__main__":
    print("Trainer module loaded")
