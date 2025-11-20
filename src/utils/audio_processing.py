import torch
import torchaudio
import soundfile as sf
import numpy as np
from scipy import signal
from pathlib import Path
from typing import Tuple, Optional


def load_audio(file_path: str, sample_rate: int = 22050, mono: bool = True) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample if needed
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate (default: 22050 Hz)
        mono: Convert to mono if True
    
    Returns:
        Tuple of (audio tensor, sample rate)
    """
    # Use soundfile instead of torchaudio.load to avoid ffmpeg hangs on Jetson
    try:
        audio_data, sr = sf.read(file_path, dtype='float32', always_2d=True)
        # Convert to torch tensor and transpose to (channels, samples)
        waveform = torch.from_numpy(audio_data.T)
    except Exception as e:
        print(f"Error loading {file_path} with soundfile: {e}")
        # Fallback to torchaudio
        waveform, sr = torchaudio.load(file_path)
    
    # Convert to mono if requested
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate
    
    return waveform, sr


def save_audio(file_path: str, audio: torch.Tensor, sample_rate: int = 22050):
    """
    Save audio tensor to file
    
    Args:
        file_path: Output file path
        audio: Audio tensor (channels, samples)
        sample_rate: Sample rate of audio
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(file_path, audio.cpu(), sample_rate)


def normalize_audio(audio: torch.Tensor, target_db: float = -20.0) -> torch.Tensor:
    """
    Normalize audio to target dB level
    
    Args:
        audio: Audio tensor
        target_db: Target loudness in dB
    
    Returns:
        Normalized audio tensor
    """
    # Calculate current RMS
    rms = torch.sqrt(torch.mean(audio ** 2))
    
    if rms == 0:
        return audio
    
    # Convert target dB to linear scale
    target_rms = 10 ** (target_db / 20)
    
    # Apply gain
    gain = target_rms / rms
    normalized = audio * gain
    
    # Prevent clipping
    max_val = torch.max(torch.abs(normalized))
    if max_val > 1.0:
        normalized = normalized / max_val
    
    return normalized


def apply_highpass_filter(audio: torch.Tensor, sample_rate: int, cutoff_freq: float = 80.0) -> torch.Tensor:
    """
    Apply highpass filter to remove low-frequency rumble
    
    Args:
        audio: Audio tensor
        sample_rate: Sample rate
        cutoff_freq: Cutoff frequency in Hz
    
    Returns:
        Filtered audio tensor
    """
    # Create highpass filter
    highpass = torchaudio.transforms.Highpass(sample_rate=sample_rate, cutoff_freq=cutoff_freq)
    return highpass(audio)


def add_noise(audio: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
    """
    Add Gaussian noise to audio for data augmentation
    
    Args:
        audio: Clean audio tensor
        noise_level: Standard deviation of noise
    
    Returns:
        Noisy audio tensor
    """
    noise = torch.randn_like(audio) * noise_level
    return audio + noise


def simulate_vinyl_artifacts(
    audio: torch.Tensor,
    sample_rate: int,
    impulse_rate: float = 10.0,
    impulse_amplitude: Tuple[float, float] = (0.1, 0.5),
    surface_noise_level: Tuple[float, float] = (0.015, 0.03),
    crackle_level: Tuple[float, float] = (0.01, 0.02),
    add_rumble: bool = True,
    add_rolloff: bool = True
) -> torch.Tensor:
    """
    Simulate shellac 78rpm record artifacts (surface noise, crackles, pops, wear) for training data
    More realistic simulation with multiple types of degradation
    
    Args:
        audio: Clean audio tensor (channels, samples)
        sample_rate: Sample rate
        impulse_rate: Average number of impulses per second
        impulse_amplitude: Min and max amplitude for impulses (0-1 scale)
        surface_noise_level: Min and max for continuous surface noise
        crackle_level: Min and max for high-frequency crackle
        add_rumble: Whether to add low-frequency rumble
        add_rolloff: Whether to add high-frequency roll-off
    
    Returns:
        Audio with simulated 78rpm artifacts
    """
    audio_with_artifacts = audio.clone()
    num_samples = audio.shape[-1]
    duration = num_samples / sample_rate
    
    # 1. Surface noise (continuous background noise) - more prominent than vinyl
    surface_noise = torch.randn_like(audio) * np.random.uniform(*surface_noise_level)
    audio_with_artifacts = audio_with_artifacts + surface_noise
    
    # 2. Random loud pops (impulses) - characteristic of 78s
    # Use Poisson distribution for more realistic timing
    expected_pops = int(duration * impulse_rate)
    num_pops = np.random.poisson(expected_pops)
    
    if num_pops > 0:
        pop_locations = np.random.randint(0, num_samples, num_pops)
        pop_amplitudes = np.random.uniform(*impulse_amplitude, num_pops)
        
        # Some impulses are asymmetric (more positive or negative)
        pop_polarities = np.random.choice([-1, 1], num_pops, p=[0.45, 0.55])
        
        for loc, amp, polarity in zip(pop_locations, pop_amplitudes, pop_polarities):
            if loc < num_samples:
                # Create a realistic impulse shape with fast attack and exponential decay
                # Longer decay for louder pops
                decay_time = np.random.uniform(0.001, 0.003) * (1 + amp)  # 1-3ms base + amplitude dependent
                decay_length = min(int(sample_rate * decay_time), num_samples - loc)
                
                if decay_length > 0:
                    # Exponential decay
                    decay = np.exp(-np.arange(decay_length) / (sample_rate * decay_time * 0.3))
                    
                    # Add some high-frequency content (clicks have sharp transients)
                    impulse = amp * polarity * decay
                    
                    # Add slight ringing (resonance)
                    if decay_length > 10:
                        resonance_freq = np.random.uniform(3000, 8000)  # High-freq resonance
                        t = np.arange(decay_length) / sample_rate
                        resonance = 0.3 * np.sin(2 * np.pi * resonance_freq * t) * decay
                        impulse = impulse + resonance * amp * 0.2
                    
                    audio_with_artifacts[..., loc:loc+decay_length] += torch.from_numpy(impulse).float().to(audio.device)
    
    # 3. Crackle (high-frequency noise) - more aggressive
    crackle = torch.randn_like(audio) * np.random.uniform(*crackle_level)
    # Apply high-pass filter using scipy butter filter
    crackle_np = crackle.cpu().numpy()
    nyquist = sample_rate / 2
    cutoff = 2500 / nyquist  # Lower cutoff for more prominent crackle
    b, a = signal.butter(4, cutoff, btype='high')
    for i in range(crackle_np.shape[0]):
        crackle_np[i] = signal.filtfilt(b, a, crackle_np[i])
    crackle = torch.from_numpy(crackle_np).to(audio.device)
    audio_with_artifacts = audio_with_artifacts + crackle
    
    # 4. Low-frequency rumble (mechanical noise)
    if add_rumble:
        rumble = torch.randn_like(audio) * np.random.uniform(0.005, 0.015)
        # Apply low-pass filter
        rumble_np = rumble.cpu().numpy()
        rumble_cutoff = 100 / nyquist  # 100 Hz rumble
        b_rumble, a_rumble = signal.butter(4, rumble_cutoff, btype='low')
        for i in range(rumble_np.shape[0]):
            rumble_np[i] = signal.filtfilt(b_rumble, a_rumble, rumble_np[i])
        rumble = torch.from_numpy(rumble_np).to(audio.device)
        audio_with_artifacts = audio_with_artifacts + rumble
    
    # 5. High-frequency roll-off (78s lose high end due to recording/playback limitations)
    if add_rolloff:
        # Apply gentle low-pass filter to simulate bandwidth limitation
        audio_np = audio_with_artifacts.cpu().numpy()
        rolloff_freq = np.random.uniform(6000, 8000) / nyquist  # Variable roll-off
        b_roll, a_roll = signal.butter(3, rolloff_freq, btype='low')
        for i in range(audio_np.shape[0]):
            audio_np[i] = signal.filtfilt(b_roll, a_roll, audio_np[i])
        audio_with_artifacts = torch.from_numpy(audio_np).to(audio.device)
    
    return audio_with_artifacts


def chunk_audio(audio: torch.Tensor, chunk_size: int, overlap: int = 0) -> list:
    """
    Split audio into chunks for processing
    
    Args:
        audio: Audio tensor (channels, samples)
        chunk_size: Size of each chunk in samples
        overlap: Overlap between chunks in samples
    
    Returns:
        List of audio chunks
    """
    chunks = []
    num_samples = audio.shape[-1]
    stride = chunk_size - overlap
    
    for start in range(0, num_samples - chunk_size + 1, stride):
        end = start + chunk_size
        chunks.append(audio[..., start:end])
    
    # Add final chunk if there are remaining samples
    if num_samples % stride != 0:
        chunks.append(audio[..., -chunk_size:])
    
    return chunks


if __name__ == "__main__":
    # Example usage
    print("Audio processing utilities loaded")
    
    # Create a test signal
    sample_rate = 22050
    duration = 2.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    test_audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)  # 440 Hz sine wave
    
    print(f"Test audio shape: {test_audio.shape}")
    
    # Test normalization
    normalized = normalize_audio(test_audio)
    print(f"Normalized audio RMS: {torch.sqrt(torch.mean(normalized ** 2)):.4f}")
    
    # Test adding artifacts
    with_artifacts = simulate_vinyl_artifacts(test_audio, sample_rate)
    print(f"Audio with artifacts shape: {with_artifacts.shape}")
