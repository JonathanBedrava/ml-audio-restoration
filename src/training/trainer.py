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
from .stereo_losses import spectral_clustering_loss, temporal_consistency_loss


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
        test_audio_dir: Optional[str] = None,
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
            test_audio_dir: Directory containing test audio files (optional)
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
        
        # Auto-discover test audio files from directory
        self.test_audio_dir = Path(test_audio_dir) if test_audio_dir else None
        self.test_output_dir = Path(test_output_dir) if test_output_dir else None
        
        if self.test_output_dir:
            self.test_output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = writer
        
        # Loss functions
        self.criterion = nn.MSELoss()  # Time-domain loss
        self.l1_criterion = nn.L1Loss()  # Better for preserving dynamics
        
        # Multi-scale spectral loss (better for low frequencies)
        self.spectral_loss_weight = 0.5
        self.fft_sizes = [512, 1024, 2048]  # Multiple scales for frequency coverage
        
        # Stereo quality losses
        self.spectral_clustering_weight = 0.1  # Smooth panning across frequencies
        self.temporal_consistency_weight = 0.05  # Stable stereo width over time
        
        # Impulse/crackle detection loss weight (for denoiser)
        self.impulse_loss_weight = 0.3
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Mixed precision training - disable on Jetson (Orin GPU)
        # Check if this is Jetson by GPU name
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        self.use_amp = "RTX" in gpu_name or "GeForce" in gpu_name  # Only for desktop GPUs
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        if not self.use_amp:
            print("Mixed precision disabled (Jetson detected)")
        
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
    
    def _stereo_balance_loss(self, stereo_output, stereo_target):
        """
        Compute balance loss with soft constraints.
        Allows natural stereo variations but prevents extreme one-sided output.
        
        Strategy:
        1. Learn the general balance distribution from targets
        2. Only penalize output when it's MUCH more imbalanced than target
        3. Allow flexibility for creative stereo imaging
        
        Args:
            stereo_output: [batch, 2, time] stereo audio output
            stereo_target: [batch, 2, time] stereo audio target
        Returns:
            Loss value (penalizes only extreme imbalances)
        """
        # Output channel RMS
        out_left_rms = torch.sqrt((stereo_output[:, 0, :] ** 2).mean(dim=1) + 1e-8)
        out_right_rms = torch.sqrt((stereo_output[:, 1, :] ** 2).mean(dim=1) + 1e-8)
        
        # Target channel RMS
        tgt_left_rms = torch.sqrt((stereo_target[:, 0, :] ** 2).mean(dim=1) + 1e-8)
        tgt_right_rms = torch.sqrt((stereo_target[:, 1, :] ** 2).mean(dim=1) + 1e-8)
        
        # Compute balance ratios (left/right energy ratio)
        out_ratio = out_left_rms / (out_right_rms + 1e-8)
        tgt_ratio = tgt_left_rms / (tgt_right_rms + 1e-8)
        
        # Allow output to deviate from target, but not too much
        # Use smooth penalty - only penalizes large deviations
        ratio_diff = torch.log(out_ratio + 1e-8) - torch.log(tgt_ratio + 1e-8)
        
        # Soft constraint: only penalize if deviation > 0.5 (about 60/40 split)
        # This allows natural stereo variations while preventing 90/10 extremes
        threshold = 0.5  # log space, corresponds to ~1.65x ratio difference
        soft_penalty = torch.nn.functional.relu(torch.abs(ratio_diff) - threshold)
        
        return soft_penalty.mean()
    
    def _decorrelation_loss(self, stereo_output):
        """
        Very mild decorrelation loss to gently encourage stereo separation.
        Penalizes high correlation between left and right channels.
        
        Args:
            stereo_output: [batch, 2, time] stereo audio output
        Returns:
            Loss value (lower when channels are less correlated)
        """
        left = stereo_output[:, 0, :]
        right = stereo_output[:, 1, :]
        
        # Normalize for correlation computation
        left_norm = (left - left.mean(dim=1, keepdim=True)) / (left.std(dim=1, keepdim=True) + 1e-8)
        right_norm = (right - right.mean(dim=1, keepdim=True)) / (right.std(dim=1, keepdim=True) + 1e-8)
        
        # Correlation (we want this low for separation)
        correlation = (left_norm * right_norm).mean(dim=1)
        
        # Squared correlation as loss
        return (correlation ** 2).mean()
    
    def _low_frequency_centering_loss(self, stereo_output):
        """
        Encourage low frequencies (<150Hz) to be centered (mono).
        This is standard mastering practice for better bass translation and phase coherence.
        
        Args:
            stereo_output: [batch, 2, time] stereo audio output
        Returns:
            Loss value (lower when low frequencies are similar in both channels)
        """
        # Apply low-pass filter using FFT
        fft_size = 2048
        hop_length = fft_size // 4
        
        left = stereo_output[:, 0, :]
        right = stereo_output[:, 1, :]
        
        # Compute STFT for both channels
        left_stft = torch.stft(
            left,
            n_fft=fft_size,
            hop_length=hop_length,
            window=torch.hann_window(fft_size).to(stereo_output.device),
            return_complex=True
        )
        right_stft = torch.stft(
            right,
            n_fft=fft_size,
            hop_length=hop_length,
            window=torch.hann_window(fft_size).to(stereo_output.device),
            return_complex=True
        )
        
        # Get magnitude of low frequency bins
        # At 22050 Hz sample rate with 2048 FFT: each bin = 22050/2048 = ~10.77 Hz
        # 150 Hz = bin 14, so use bins 0-14
        low_freq_bins = 14
        
        left_low = torch.abs(left_stft[:, :low_freq_bins, :])
        right_low = torch.abs(right_stft[:, :low_freq_bins, :])
        
        # L1 loss between low frequency content (we want them similar)
        return self.l1_criterion(left_low, right_low)
    
    def _compute_stereo_metrics(self, stereo_output):
        """
        Compute metrics to monitor stereo separation quality
        
        Returns dict with:
        - correlation: L/R correlation (0=uncorrelated, 1=identical)
        - width: Stereo width measure (0=mono, 1=full width)
        """
        with torch.no_grad():
            left = stereo_output[:, 0, :]
            right = stereo_output[:, 1, :]
            
            # Correlation
            left_norm = (left - left.mean(dim=1, keepdim=True)) / (left.std(dim=1, keepdim=True) + 1e-8)
            right_norm = (right - right.mean(dim=1, keepdim=True)) / (right.std(dim=1, keepdim=True) + 1e-8)
            correlation = (left_norm * right_norm).mean().item()
            
            # Width measure: ratio of side to mid+side energy
            mid = (left + right) / 2.0
            side = (left - right) / 2.0
            mid_energy = (mid ** 2).mean()
            side_energy = (side ** 2).mean()
            width = (side_energy / (mid_energy + side_energy + 1e-8)).item()
            
            return {
                'correlation': abs(correlation),
                'width': width
            }
    
    def _spectral_loss(self, output, target):
        """
        Multi-scale spectral loss using STFT.
        Better preserves frequency content, especially low frequencies.
        
        Args:
            output: Predicted audio [batch, channels, time]
            target: Target audio [batch, channels, time]
        Returns:
            Spectral loss value
        """
        loss = 0.0
        
        for fft_size in self.fft_sizes:
            hop_length = fft_size // 4
            
            # Compute STFT for each channel
            for ch in range(output.shape[1]):
                output_stft = torch.stft(
                    output[:, ch, :],
                    n_fft=fft_size,
                    hop_length=hop_length,
                    window=torch.hann_window(fft_size).to(output.device),
                    return_complex=True
                )
                target_stft = torch.stft(
                    target[:, ch, :],
                    n_fft=fft_size,
                    hop_length=hop_length,
                    window=torch.hann_window(fft_size).to(target.device),
                    return_complex=True
                )
                
                # Magnitude loss (log scale emphasizes low frequencies)
                output_mag = torch.abs(output_stft)
                target_mag = torch.abs(target_stft)
                
                mag_loss = self.l1_criterion(
                    torch.log(output_mag + 1e-5),
                    torch.log(target_mag + 1e-5)
                )
                
                loss += mag_loss
        
        # Average over FFT sizes and channels
        return loss / (len(self.fft_sizes) * output.shape[1])
    
    def _impulse_loss(self, output, target):
        """
        Loss that emphasizes impulse/transient differences.
        Penalizes failure to remove crackles and pops.
        
        Args:
            output: Denoised audio [batch, channels, time]
            target: Clean audio [batch, channels, time]
        Returns:
            Loss value emphasizing impulse errors
        """
        # Compute derivatives to emphasize transient errors
        output_diff = torch.abs(output[:, :, 1:] - output[:, :, :-1])
        target_diff = torch.abs(target[:, :, 1:] - target[:, :, :-1])
        
        # Second derivative catches sharp spikes (crackles/pops)
        output_diff2 = torch.abs(output_diff[:, :, 1:] - output_diff[:, :, :-1])
        target_diff2 = torch.abs(target_diff[:, :, 1:] - target_diff[:, :, :-1])
        
        # L1 loss on second derivative (more sensitive to impulses than MSE)
        impulse_loss = self.l1_criterion(output_diff2, target_diff2)
        
        # Also penalize high-amplitude transient differences
        transient_error = torch.abs(output_diff - target_diff)
        high_energy_mask = (target_diff > target_diff.mean() * 2.0).float()
        weighted_transient_loss = (transient_error * high_energy_mask).mean()
        
        return impulse_loss + weighted_transient_loss * 0.5
    
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
            
            self.optimizer.zero_grad(set_to_none=True)  # Free memory more aggressively
            
            # Forward pass with optional mixed precision
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    output = self.model(degraded)
                    
                    # Combined loss: time-domain + spectral
                    time_loss = self.criterion(output, clean)
                    spec_loss = self._spectral_loss(output, clean)
                    recon_loss = time_loss + self.spectral_loss_weight * spec_loss
                    
                    # Add model-specific losses
                    if output.shape[1] == 1:
                        # Mono models (denoiser): add impulse loss
                        impulse_loss = self._impulse_loss(output, clean)
                        loss = recon_loss + self.impulse_loss_weight * impulse_loss
                    else:
                        # Stereo models: add spectral clustering + temporal consistency
                        clustering_loss = spectral_clustering_loss(output)
                        consistency_loss = temporal_consistency_loss(output)
                        loss = recon_loss + self.spectral_clustering_weight * clustering_loss + self.temporal_consistency_weight * consistency_loss
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular forward/backward without AMP
                output = self.model(degraded)
                
                # Combined loss: time-domain + spectral
                time_loss = self.criterion(output, clean)
                spec_loss = self._spectral_loss(output, clean)
                recon_loss = time_loss + self.spectral_loss_weight * spec_loss
                
                # Add model-specific losses
                if output.shape[1] == 1:
                    # Mono models (denoiser): add impulse loss
                    impulse_loss = self._impulse_loss(output, clean)
                    loss = recon_loss + self.impulse_loss_weight * impulse_loss
                else:
                    # Stereo models: add spectral clustering + temporal consistency
                    clustering_loss = spectral_clustering_loss(output)
                    consistency_loss = temporal_consistency_loss(output)
                    loss = recon_loss + self.spectral_clustering_weight * clustering_loss + self.temporal_consistency_weight * consistency_loss
                    
                loss.backward()
                self.optimizer.step()
            
            # Log stereo metrics before deleting output (if logging this batch)
            if self.writer and batch_idx % 50 == 0 and output.shape[1] == 2:
                metrics = self._compute_stereo_metrics(output)
                self.writer.add_scalar('Stereo/correlation', metrics['correlation'], self.global_step)
                self.writer.add_scalar('Stereo/width', metrics['width'], self.global_step)
            
            # Explicitly delete tensors to free memory
            del output
            if batch_idx % 10 == 0:  # Clear cache periodically
                torch.cuda.empty_cache()
            
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
                
                # Combined loss: time-domain + spectral
                time_loss = self.criterion(output, clean)
                spec_loss = self._spectral_loss(output, clean)
                recon_loss = time_loss + self.spectral_loss_weight * spec_loss
                
                # Add model-specific losses
                if output.shape[1] == 1:
                    # Mono models (denoiser): add impulse loss
                    impulse_loss = self._impulse_loss(output, clean)
                    loss = recon_loss + self.impulse_loss_weight * impulse_loss
                else:
                    # Stereo models: add spectral clustering + temporal consistency
                    clustering_loss = spectral_clustering_loss(output)
                    consistency_loss = temporal_consistency_loss(output)
                    loss = recon_loss + self.spectral_clustering_weight * clustering_loss + self.temporal_consistency_weight * consistency_loss
                
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
            
            # Print stereo metrics if stereo model
            if hasattr(self, '_last_stereo_metrics'):
                metrics = self._last_stereo_metrics
                print(f"Stereo Correlation: {metrics['correlation']:.3f} (lower=better separation)")
                print(f"Stereo Width: {metrics['width']:.3f} (higher=more width)")
            
            # Save checkpoint
            if self.epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.epoch}.pth')
                # Generate test output if test audio directory provided
                if self.test_audio_dir:
                    self.generate_test_output(f'epoch_{self.epoch}')
            
            # Save best model
            if self.val_loader is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                print(f"✓ Saved best model with val_loss: {val_loss:.6f}")
                # Generate test output for best model
                if self.test_audio_dir:
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
            
            # Generate output (respect use_amp setting)
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    output = self.model(degraded)
            else:
                output = self.model(degraded)
            
            # Log to TensorBoard (assuming 22050 Hz sample rate)
            # For stereo, convert to mono for logging or log channels separately
            if degraded.shape[1] > 1:
                self.writer.add_audio('Audio/degraded', degraded[0][0].cpu(), self.epoch, sample_rate=22050)
            else:
                self.writer.add_audio('Audio/degraded', degraded[0].squeeze().cpu(), self.epoch, sample_rate=22050)
            
            if clean.shape[1] > 1:
                # Log stereo channels separately
                self.writer.add_audio('Audio/clean_left', clean[0][0].cpu(), self.epoch, sample_rate=22050)
                self.writer.add_audio('Audio/clean_right', clean[0][1].cpu(), self.epoch, sample_rate=22050)
            else:
                self.writer.add_audio('Audio/clean', clean[0].squeeze().cpu(), self.epoch, sample_rate=22050)
            
            if output.shape[1] > 1:
                # Log stereo channels separately
                self.writer.add_audio('Audio/restored_left', output[0][0].cpu(), self.epoch, sample_rate=22050)
                self.writer.add_audio('Audio/restored_right', output[0][1].cpu(), self.epoch, sample_rate=22050)
            else:
                self.writer.add_audio('Audio/restored', output[0].squeeze().cpu(), self.epoch, sample_rate=22050)
    
    def generate_test_output(self, suffix: str):
        """Generate denoised output from all test audio files in directory (first 30 seconds each)"""
        if not self.test_audio_dir or not self.test_output_dir:
            return
        
        import soundfile as sf
        import numpy as np
        
        # Find all audio files in test directory root (not subdirectories)
        test_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            test_files.extend(self.test_audio_dir.glob(ext))
        
        if not test_files:
            print(f"  No test audio files found in {self.test_audio_dir}")
            return
        
        print(f"  Generating test outputs for {len(test_files)} files: {suffix}")
        
        self.model.eval()
        with torch.no_grad():
            for test_file in test_files:
                # Use file stem (without extension) as identifier
                file_id = test_file.stem
                print(f"    Processing: {test_file.name}")
                
                # Load test audio using soundfile (same as training)
                waveform_np, sample_rate = sf.read(str(test_file), always_2d=True)
                # Force contiguous memory layout before creating tensor (fixes cuDNN LSTM error)
                waveform = torch.from_numpy(np.ascontiguousarray(waveform_np.T)).float()  # [channels, samples]
                
                # Save original audio only once (not every time)
                original_path = self.test_output_dir / f'{file_id}_original.wav'
                if not original_path.exists():
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
                
                # Resample to 22050 if needed
                if sample_rate != 22050:
                    import librosa
                    waveform_np = waveform.squeeze(0).numpy()
                    waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=22050)
                    # Ensure contiguous after resampling
                    waveform = torch.from_numpy(np.ascontiguousarray(waveform_np)).unsqueeze(0)
                    sample_rate = 22050
                
                # Take only first 30 seconds
                max_samples = 22050 * 30  # 30 seconds at 22050 Hz
                if waveform.shape[1] > max_samples:
                    waveform = waveform[:, :max_samples]
                
                # Move to device
                waveform = waveform.to(self.device)
                
                # Process in chunks - use same 2s chunks as training to avoid cuDNN LSTM sequence length issues
                chunk_size = 22050 * 2  # 2 second chunks (same as training)
                output_chunks = []
                
                for i in range(0, waveform.shape[1], chunk_size):
                    chunk = waveform[:, i:i+chunk_size].contiguous()
                    
                    # Pad if necessary
                    if chunk.shape[1] < chunk_size:
                        padding = chunk_size - chunk.shape[1]
                        chunk = torch.nn.functional.pad(chunk, (0, padding)).contiguous()
                        had_padding = True
                    else:
                        had_padding = False
                    
                    # Add batch dimension and ensure contiguous
                    chunk = chunk.unsqueeze(0).contiguous()
                    
                    # Process
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
                
                # Save degraded audio (mono)
                degraded_path = self.test_output_dir / f'{file_id}_degraded_{suffix}.wav'
                sf.write(
                    str(degraded_path),
                    waveform_cpu.squeeze(0).numpy(),
                    sample_rate
                )
                
                # Save restored audio (stereo) - transpose for soundfile format
                output_path = self.test_output_dir / f'{file_id}_restored_{suffix}.wav'
                sf.write(
                    str(output_path),
                    restored.numpy().T,  # Transpose to (samples, channels)
                    sample_rate
                )
                
                # Cleanup: keep only 'best' and most recent epoch (delete old epochs)
                if suffix.startswith('epoch_'):
                    current_epoch = int(suffix.split('_')[1])
                    
                    # Find all epoch files for this file_id
                    epoch_files = []
                    for pattern in [f'{file_id}_restored_epoch_*.wav', f'{file_id}_degraded_epoch_*.wav']:
                        epoch_files.extend(self.test_output_dir.glob(pattern))
                    
                    # Delete all epoch files except the current one
                    for f in epoch_files:
                        try:
                            file_epoch = int(f.stem.split('_epoch_')[1])
                            if file_epoch != current_epoch:
                                f.unlink()
                        except:
                            pass  # Skip files we can't parse
        
        print(f"  ✓ Test outputs saved to: {self.test_output_dir}")

    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint and cleanup old ones"""
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
        
        # Cleanup: keep only 'best' and most recent epoch checkpoint
        if filename.startswith('checkpoint_epoch_'):
            # Delete all other epoch checkpoints
            for ckpt in self.checkpoint_dir.glob('checkpoint_epoch_*.pth'):
                if ckpt != checkpoint_path:  # Don't delete the one we just saved
                    ckpt.unlink()
                    print(f"  Deleted old checkpoint: {ckpt.name}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load state dict directly (no conversion needed)
        state_dict = checkpoint['model_state_dict']
        
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}")


if __name__ == "__main__":
    print("Trainer module loaded")
