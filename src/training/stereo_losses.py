"""
Additional stereo losses for temporal consistency and spectral clustering
"""
import torch


def spectral_clustering_loss(stereo_output):
    """
    Encourage similar frequencies to have similar stereo placement.
    This ensures instrument coherence - if 500Hz is panned left, 550Hz should be too.
    
    Args:
        stereo_output: [batch, 2, time] stereo audio output
    Returns:
        Loss value (lower when similar frequencies have similar panning)
    """
    fft_size = 2048
    hop_length = fft_size // 4
    
    left = stereo_output[:, 0, :]
    right = stereo_output[:, 1, :]
    
    # Compute STFT
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
    
    # Compute stereo position for each frequency bin: (L-R)/(L+R+eps)
    # Range: -1 (hard left) to +1 (hard right), 0 = center
    left_mag = torch.abs(left_stft)
    right_mag = torch.abs(right_stft)
    stereo_position = (left_mag - right_mag) / (left_mag + right_mag + 1e-8)
    
    # Penalize large differences between adjacent frequency bins
    # If bin N is panned left, bin N+1 shouldn't be hard right
    position_diff = torch.diff(stereo_position, dim=1)  # Diff along frequency axis
    
    # Use smooth L1 (Huber) loss - small differences OK, large jumps penalized
    return torch.nn.functional.smooth_l1_loss(position_diff, torch.zeros_like(position_diff))


def temporal_consistency_loss(stereo_output):
    """
    Encourage stereo width to be consistent over time.
    Prevents chaotic jumping of stereo image frame-to-frame.
    
    Args:
        stereo_output: [batch, 2, time] stereo audio output
    Returns:
        Loss value (lower when stereo width is temporally smooth)
    """
    # Compute instantaneous stereo width in short windows
    window_size = 512
    hop = 256
    
    left = stereo_output[:, 0, :]
    right = stereo_output[:, 1, :]
    
    # Unfold into overlapping windows
    left_windows = left.unfold(-1, window_size, hop)  # [batch, num_windows, window_size]
    right_windows = right.unfold(-1, window_size, hop)
    
    # Compute RMS energy per window
    left_rms = torch.sqrt((left_windows ** 2).mean(dim=-1) + 1e-8)
    right_rms = torch.sqrt((right_windows ** 2).mean(dim=-1) + 1e-8)
    
    # Compute width measure: side_energy / (mid_energy + side_energy)
    mid = (left_rms + right_rms) / 2
    side = torch.abs(left_rms - right_rms) / 2
    width = side / (mid + side + 1e-8)
    
    # Penalize large changes in width between adjacent windows
    width_diff = torch.diff(width, dim=-1)
    
    # Use L2 loss - we want smooth transitions
    return (width_diff ** 2).mean()
