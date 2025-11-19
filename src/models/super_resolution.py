import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioSuperResolution(nn.Module):
    """
    Audio Super-Resolution model for bandwidth extension.
    Reconstructs high-frequency content and transients that were lost
    due to degradation or limited recording bandwidth.
    
    Upsamples from 22.05kHz to 44.1kHz (or higher) and regenerates
    high-frequency harmonics and transients.
    """
    
    def __init__(
        self,
        upscale_factor: int = 2,  # 2x = 22.05kHz -> 44.1kHz
        channels: int = 1,
        base_channels: int = 64,
        num_residual_blocks: int = 8
    ):
        super(AudioSuperResolution, self).__init__()
        
        self.upscale_factor = upscale_factor
        
        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv1d(channels, base_channels, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        # Residual blocks for feature learning
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(base_channels) for _ in range(num_residual_blocks)
        ])
        
        # Middle convolution
        self.middle = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels)
        )
        
        # Upsampling layers (subpixel convolution approach)
        self.upsample_blocks = nn.ModuleList()
        current_channels = base_channels
        
        # Add upsampling blocks based on upscale factor
        num_upsample = int(torch.log2(torch.tensor(upscale_factor)).item())
        for _ in range(num_upsample):
            self.upsample_blocks.append(
                UpsampleBlock(current_channels, current_channels * 2)
            )
        
        # High-frequency emphasis network
        self.hf_emphasis = nn.Sequential(
            nn.Conv1d(current_channels, base_channels, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=5, padding=2),
            nn.PReLU(),
        )
        
        # Transient enhancement branch
        self.transient_enhancer = nn.Sequential(
            nn.Conv1d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(base_channels // 2, base_channels // 2, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        
        # Final reconstruction
        self.reconstruction = nn.Conv1d(
            base_channels + base_channels // 2, channels, kernel_size=9, padding=4
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Low-resolution audio tensor (batch, channels, samples)
        
        Returns:
            High-resolution audio with restored high frequencies
        """
        # Initial feature extraction
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
        hf_features = self.hf_emphasis(features)
        
        # Transient enhancement
        transient_features = self.transient_enhancer(hf_features)
        
        # Combine and reconstruct
        combined = torch.cat([hf_features, transient_features], dim=1)
        output = self.reconstruction(combined)
        
        # Residual connection with upsampled input
        upsampled_input = F.interpolate(
            x, scale_factor=self.upscale_factor, mode='linear', align_corners=False
        )
        output = output + upsampled_input
        
        return output


class ResidualBlock(nn.Module):
    """Residual block for feature learning"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual


class UpsampleBlock(nn.Module):
    """Upsampling block using subpixel convolution"""
    
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)  # 2x upsampling
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        # Reshape for pixel shuffle
        batch, channels, length = x.shape
        x = self.conv(x)
        
        # Rearrange for 1D pixel shuffle
        # Convert (B, C, L) to (B, C, 1, L) for 2D pixel shuffle
        x = x.unsqueeze(2)
        # Apply pixel shuffle (reduces channels, increases spatial dims)
        # This gives us (B, C//2, 2, L)
        x = self.pixel_shuffle(x)
        # Reshape back: (B, C//2, 2*L)
        x = x.squeeze(2)
        
        # For 1D, we manually implement the shuffle
        batch, channels, length = self.conv(x.squeeze(2) if x.dim() == 4 else x).shape
        x_reshaped = self.conv(x.squeeze(2) if x.dim() == 4 else x)
        x_upsampled = F.interpolate(x_reshaped, scale_factor=2, mode='linear', align_corners=False)
        
        return self.prelu(x_upsampled)


class SpectralLoss(nn.Module):
    """
    Spectral loss for training super-resolution models.
    Compares spectrograms in addition to time-domain MSE.
    """
    
    def __init__(self, n_fft=2048, hop_length=512, alpha=0.5):
        super(SpectralLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha = alpha  # Balance between time and frequency domain loss
    
    def forward(self, output, target):
        # Time domain loss
        time_loss = F.mse_loss(output, target)
        
        # Frequency domain loss
        output_spec = torch.stft(
            output.squeeze(1), 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            return_complex=True
        )
        target_spec = torch.stft(
            target.squeeze(1), 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            return_complex=True
        )
        
        # Magnitude loss
        spec_loss = F.mse_loss(
            torch.abs(output_spec), 
            torch.abs(target_spec)
        )
        
        # Combined loss
        total_loss = self.alpha * time_loss + (1 - self.alpha) * spec_loss
        
        return total_loss


if __name__ == "__main__":
    # Test the model
    model = AudioSuperResolution(upscale_factor=2)
    
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
