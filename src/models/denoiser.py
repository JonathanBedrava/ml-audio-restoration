import torch
import torch.nn as nn


class AudioDenoiser(nn.Module):
    """
    U-Net based audio denoising model for removing noise, crackles, and pops
    from 78rpm record recordings.
    """
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(AudioDenoiser, self).__init__()
        
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
        
        # Final output layer
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)
    
    def _conv_block(self, in_channels, out_channels):
        """Create a convolutional block with batch norm and ReLU"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        """
        Forward pass through the denoiser
        
        Args:
            x: Input audio tensor of shape (batch, channels, samples)
        
        Returns:
            Denoised audio tensor
        """
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
        
        return self.final_conv(x)


if __name__ == "__main__":
    # Test the model
    model = AudioDenoiser()
    x = torch.randn((1, 1, 16000))  # 1 second at 16kHz
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
