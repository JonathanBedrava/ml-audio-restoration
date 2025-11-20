import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioSuperResolution(nn.Module):
    """
    Audio super-resolution for bandwidth extension.
    Optimized for Jetson with ~100K parameters.
    """
    
    def __init__(
        self,
        upscale_factor: int = 2,
        channels: int = 1,
        base_channels: int = 32,
        num_residual_blocks: int = 4
    ):
        super().__init__()
        
        self.upscale_factor = upscale_factor
        
        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv1d(channels, base_channels, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Residual blocks for feature learning
        self.residual_blocks = nn.ModuleList([
            ResidualBlockEfficient(base_channels) for _ in range(num_residual_blocks)
        ])
        
        # Middle convolution
        self.middle = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels)
        )
        
        # Transposed conv upsampling (simpler than subpixel)
        num_upsample = int(torch.log2(torch.tensor(upscale_factor)).item())
        self.upsample_blocks = nn.ModuleList()
        
        for _ in range(num_upsample):
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        base_channels, base_channels,
                        kernel_size=4, stride=2, padding=1
                    ),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        
        # High-frequency emphasis (lighter than original)
        self.hf_emphasis = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final reconstruction - no activation to avoid limiting output range
        self.reconstruction = nn.Conv1d(
            base_channels, channels, kernel_size=7, padding=3
        )
    
    def forward(self, x):
        """
        Args:
            x: Low-resolution audio [batch, channels, samples]
        Returns:
            High-resolution audio [batch, channels, samples * upscale_factor]
        """
        # Initial features
        initial_features = self.initial(x)
        
        # Residual learning
        residual = initial_features
        for block in self.residual_blocks:
            residual = block(residual)
        
        # Skip connection
        residual = self.middle(residual)
        features = initial_features + residual
        
        # Upsample
        for upsample_block in self.upsample_blocks:
            features = upsample_block(features)
        
        # High-frequency emphasis
        features = self.hf_emphasis(features)
        
        # Reconstruct
        output = self.reconstruction(features)
        
        # Residual connection with upsampled input
        upsampled_input = F.interpolate(
            x, scale_factor=self.upscale_factor, mode='linear', align_corners=False
        )
        output = output + upsampled_input
        
        return output


class ResidualBlockEfficient(nn.Module):
    """Lightweight residual block"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual


class SpectralLoss(nn.Module):
    """
    Multi-scale spectral loss with transient preservation.
    Balances frequency reconstruction with sharp temporal events.
    """
    
    def __init__(self, fft_sizes=[512, 1024, 2048], alpha=0.3, transient_weight=0.3):
        super(SpectralLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.alpha = alpha  # Balance between time and frequency domain
        self.transient_weight = transient_weight  # Weight for transient loss
        self.l1_loss = nn.L1Loss()
    
    def _detect_transients(self, audio):
        """
        Detect transient regions using envelope differentiation.
        Returns binary mask with 1.0 at transient locations.
        """
        # Compute envelope using absolute value
        envelope = torch.abs(audio)
        
        # Smooth envelope slightly
        kernel_size = 64
        kernel = torch.ones(1, 1, kernel_size, device=audio.device) / kernel_size
        envelope_smooth = F.conv1d(
            envelope.unsqueeze(1) if envelope.dim() == 2 else envelope,
            kernel,
            padding=kernel_size // 2
        ).squeeze(1)
        
        # Compute derivative (rate of change)
        diff = torch.abs(envelope_smooth[:, 1:] - envelope_smooth[:, :-1])
        diff = F.pad(diff, (0, 1))  # Pad to match original length
        
        # Threshold to find transients (top 10% of changes)
        threshold = torch.quantile(diff, 0.9, dim=-1, keepdim=True)
        transient_mask = (diff > threshold).float()
        
        # Dilate transient regions slightly to cover full attack
        kernel_dilate = torch.ones(1, 1, 128, device=audio.device)
        transient_mask = F.conv1d(
            transient_mask.unsqueeze(1),
            kernel_dilate,
            padding=64
        ).squeeze(1).clamp(0, 1)
        
        return transient_mask
    
    def forward(self, output, target):
        """
        Args:
            output: Predicted audio [batch, channels, time]
            target: Target audio [batch, channels, time]
        """
        # Time domain loss (basic reconstruction)
        time_loss = F.mse_loss(output, target)
        
        # Transient-aware time loss (emphasizes sharp events)
        transient_mask = self._detect_transients(target[:, 0, :])  # Use first channel
        transient_mask = transient_mask.unsqueeze(1)  # [batch, 1, time]
        
        # Higher weight on transient regions
        weighted_diff = torch.abs(output - target)
        transient_loss = (weighted_diff * transient_mask).mean()
        steady_loss = (weighted_diff * (1 - transient_mask)).mean()
        transient_time_loss = transient_loss * 2.0 + steady_loss  # 2x weight on transients
        
        # Multi-scale frequency domain loss
        spec_loss = 0.0
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
                
                # Log-magnitude loss (emphasizes all frequencies evenly)
                output_mag = torch.abs(output_stft)
                target_mag = torch.abs(target_stft)
                
                mag_loss = self.l1_loss(
                    torch.log(output_mag + 1e-5),
                    torch.log(target_mag + 1e-5)
                )
                
                spec_loss += mag_loss
        
        # Average over FFT sizes and channels
        spec_loss = spec_loss / (len(self.fft_sizes) * output.shape[1])
        
        # Combined loss: balance spectral smoothness with transient sharpness
        total_loss = (
            self.alpha * time_loss +
            self.transient_weight * transient_time_loss +
            (1 - self.alpha - self.transient_weight) * spec_loss
        )
        
        return total_loss


if __name__ == "__main__":
    # Test the model
    model = AudioSuperResolution(upscale_factor=2, base_channels=32, num_residual_blocks=4)
    
    # Simulate 22.05kHz input (1 second)
    x = torch.randn((2, 1, 22050))
    
    output = model(x)
    print(f"Input shape: {x.shape} (22.05kHz)")
    print(f"Output shape: {output.shape} (44.1kHz)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test spectral loss
    loss_fn = SpectralLoss()
    target = torch.randn_like(output)
    loss = loss_fn(output, target)
    print(f"Spectral loss: {loss.item():.6f}")
