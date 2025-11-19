import torch
import torch.nn as nn


class StereoSeparator(nn.Module):
    """
    Model for converting mono audio to stereo with spatial separation.
    Uses a dual-branch architecture to generate left and right channels
    with distinct spatial characteristics.
    """
    
    def __init__(self, input_channels=1, hidden_dim=512, num_layers=4):
        super(StereoSeparator, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        # Shared encoder to extract features from mono input
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            256, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True
        )
        
        # Separate decoders for left and right channels
        self.left_decoder = self._create_decoder()
        self.right_decoder = self._create_decoder()
        
        # Additional spatial processing layers
        self.spatial_processor = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 2, kernel_size=5, padding=2),
        )
    
    def _create_decoder(self):
        """Create decoder branch for one channel"""
        return nn.Sequential(
            nn.Conv1d(1024, 256, kernel_size=7, padding=3),  # 1024 = hidden_dim * 2 (bidirectional)
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        Forward pass to generate stereo from mono
        
        Args:
            x: Mono audio tensor of shape (batch, 1, samples)
        
        Returns:
            Stereo audio tensor of shape (batch, 2, samples)
        """
        batch_size, _, seq_len = x.shape
        
        # Encode mono input
        features = self.encoder(x)
        
        # Reshape for LSTM (batch, seq, features)
        features = features.permute(0, 2, 1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Reshape back (batch, features, seq)
        lstm_out = lstm_out.permute(0, 2, 1)
        
        # Generate left and right channels
        left_channel = self.left_decoder(lstm_out)
        right_channel = self.right_decoder(lstm_out)
        
        # Combine channels
        stereo = torch.cat([left_channel, right_channel], dim=1)
        
        # Apply spatial processing to enhance separation
        stereo = self.spatial_processor(stereo)
        
        return stereo


if __name__ == "__main__":
    # Test the model
    model = StereoSeparator()
    x = torch.randn((2, 1, 16000))  # Batch of 2, mono, 1 second at 16kHz
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
