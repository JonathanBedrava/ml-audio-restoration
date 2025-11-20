import torch
import torch.nn as nn


class StereoSeparator(nn.Module):
    """
    Stereo separator using dilated convs + LSTM
    Optimized for Jetson with ~500K parameters
    """
    
    def __init__(
        self,
        base_channels: int = 32,
        lstm_hidden: int = 64,
        num_lstm_layers: int = 1
    ):
        super().__init__()
        
        self.lstm_hidden = lstm_hidden
        
        # Dilated conv encoder (builds receptive field efficiently)
        self.encoder = nn.ModuleList([
            # Initial conv
            nn.Sequential(
                nn.Conv1d(1, base_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(base_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Dilated blocks (dilation = 2^i)
            self._dilated_block(base_channels, base_channels * 2, dilation=1),
            self._dilated_block(base_channels * 2, base_channels * 4, dilation=2),
            self._dilated_block(base_channels * 4, base_channels * 4, dilation=4),
            self._dilated_block(base_channels * 4, base_channels * 4, dilation=8),
        ])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=base_channels * 4,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=False  # Unidirectional to save memory
        )
        
        # Stereo decoders (shared architecture, separate instances)
        self.left_decoder = self._build_decoder(lstm_hidden, base_channels)
        self.right_decoder = self._build_decoder(lstm_hidden, base_channels)
        
    def _dilated_block(self, in_channels, out_channels, dilation):
        """Residual dilated conv block"""
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                dilation=dilation,
                padding=dilation
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def _build_decoder(self, lstm_hidden, base_channels):
        """Build decoder for one channel"""
        return nn.Sequential(
            nn.Conv1d(lstm_hidden, base_channels * 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(base_channels * 4, base_channels * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(base_channels * 2, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(base_channels, 1, kernel_size=7, padding=3)
            # No activation - let network learn natural audio range
        )
    
    def forward(self, x):
        """
        Args:
            x: Input mono audio [batch, 1, time]
        Returns:
            Stereo output [batch, 2, time]
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Encode with dilated convs
        features = x
        for layer in self.encoder:
            features = layer(features)  # [batch, channels, time]
        
        # Ensure contiguous before permute
        features = features.contiguous()
        
        # Prepare for LSTM (swap dimensions)
        features = features.permute(0, 2, 1).contiguous()  # [batch, time, channels]
        
        # LSTM temporal modeling (no hidden state passed between batches)
        lstm_out, _ = self.lstm(features)  # [batch, time, lstm_hidden]
        
        # Ensure contiguous before permute back
        lstm_out = lstm_out.contiguous()
        
        # Swap back for conv
        lstm_out = lstm_out.permute(0, 2, 1).contiguous()  # [batch, lstm_hidden, time]
        
        # Decode to stereo channels
        left = self.left_decoder(lstm_out)
        right = self.right_decoder(lstm_out)
        
        # Combine channels
        stereo = torch.cat([left, right], dim=1)  # [batch, 2, time]
        
        return stereo


if __name__ == "__main__":
    # Test model
    model = StereoSeparator(base_channels=32, lstm_hidden=64)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 22050)  # 2 samples, 1 second at 22kHz
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
