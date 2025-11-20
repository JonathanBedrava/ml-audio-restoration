import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioDenoiser(nn.Module):
    """
    Efficient U-Net based audio denoising with specialized crackle/pop removal.
    Combines spectral denoising with transient impulse detection.
    Optimized for Jetson with reduced feature channels.
    """
    
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128]):
        super().__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Encoder (downsampling path)
        for feature in features:
            self.encoder.append(self._conv_block(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        
        # Decoder (upsampling path)
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose1d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.decoder.append(self._conv_block(feature * 2, feature))
        
        # Transient/impulse detection branch for crackles and pops
        # Uses small kernels to detect sharp, localized artifacts
        self.transient_detector = nn.Sequential(
            nn.Conv1d(features[0], features[0] // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(features[0] // 2, features[0] // 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(features[0] // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output as mask (0-1)
        )
        
        # Final reconstruction - combines U-Net output with transient suppression
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)
    
    def _conv_block(self, in_channels, out_channels):
        """Create a convolutional block with batch norm and LeakyReLU"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def _detect_impulses(self, x):
        """
        Detect impulse artifacts (crackles/pops) using derivative analysis.
        Returns a soft mask indicating impulse locations.
        """
        # Compute first derivative (rate of change)
        diff = torch.abs(x[:, :, 1:] - x[:, :, :-1])
        diff = F.pad(diff, (0, 1))  # Pad to match original length
        
        # Compute second derivative (acceleration - catches sharp spikes)
        diff2 = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
        diff2 = F.pad(diff2, (0, 1))
        
        # Combine with amplitude to catch loud pops
        amplitude = torch.abs(x)
        
        # Create impulse score: high derivative + high amplitude = impulse
        impulse_score = (diff2 * 2.0 + diff + amplitude * 0.5) / 3.5
        
        # Smooth the score slightly to avoid over-suppression
        kernel_size = 5
        kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
        impulse_score = F.conv1d(impulse_score, kernel, padding=kernel_size // 2)
        
        return impulse_score.clamp(0, 1)
    
    def forward(self, x):
        """
        Forward pass through the denoiser
        
        Args:
            x: Input audio tensor [batch, channels, samples]
        
        Returns:
            Denoised audio tensor [batch, channels, samples]
        """
        # Store original for residual connection
        input_audio = x
        
        skip_connections = []
        
        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            # Handle size mismatches
            if x.shape != skip_connection.shape:
                x = nn.functional.pad(x, [0, skip_connection.shape[2] - x.shape[2]])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)
        
        # Detect impulses in the decoded features
        transient_mask = self.transient_detector(x)
        
        # Also detect impulses in input (for very strong artifacts)
        input_impulse_mask = self._detect_impulses(input_audio)
        
        # Combine both detection methods
        combined_mask = torch.maximum(transient_mask, input_impulse_mask)
        
        # Reconstruct audio
        denoised = self.final_conv(x)
        
        # Suppress detected impulses: reduce magnitude at impulse locations
        # Invert mask so 1 = keep, 0 = suppress
        suppression_mask = 1.0 - combined_mask * 0.9  # Suppress up to 90%
        denoised = denoised * suppression_mask
        
        return denoised


if __name__ == "__main__":
    # Test the model
    model = AudioDenoiser()
    x = torch.randn((1, 1, 16000))  # 1 second at 16kHz
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
