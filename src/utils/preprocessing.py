import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import Optional, Tuple
from .audio_processing import load_audio, normalize_audio, simulate_vinyl_artifacts


class AudioRestorationDataset(Dataset):
    """
    Dataset for audio restoration tasks
    Pairs clean audio with degraded versions for training
    """
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 22050,
        chunk_duration: float = 2.0,
        add_artifacts: bool = True,
        transform: Optional[callable] = None
    ):
        """
        Args:
            data_dir: Directory containing audio files
            sample_rate: Target sample rate
            chunk_duration: Duration of audio chunks in seconds
            add_artifacts: Whether to add vinyl artifacts to create training pairs
            transform: Optional transform to apply to audio
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.add_artifacts = add_artifacts
        self.transform = transform
        
        # Find all audio files
        self.audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            self.audio_files.extend(self.data_dir.glob(f"**/{ext}"))
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")
        
        print(f"Found {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a pair of (degraded_audio, clean_audio)
        """
        audio_path = self.audio_files[idx]
        
        # Load audio efficiently using chunked loading (like StereoDataset)
        import soundfile as sf
        try:
            # Get file info first to check duration
            file_info = sf.info(str(audio_path))
            total_frames = file_info.frames
            
            # If file is longer than chunk, load only a random chunk
            if total_frames > self.chunk_size:
                # Pick random start position
                max_start = total_frames - self.chunk_size
                start_frame = np.random.randint(0, max_start + 1)
                
                # Load only the chunk we need (MUCH more memory efficient!)
                audio_data, sr = sf.read(
                    str(audio_path),
                    start=start_frame,
                    frames=self.chunk_size,
                    dtype='float32',
                    always_2d=True
                )
                # Convert to mono if stereo
                if audio_data.shape[1] > 1:
                    audio_data = audio_data.mean(axis=1, keepdims=True)
                audio = torch.from_numpy(audio_data.T)  # Shape: (1, samples)
            else:
                # File is short, load whole thing
                audio, sr = load_audio(str(audio_path), self.sample_rate, mono=True)
        except Exception as e:
            # Fallback to old method if something fails
            print(f"Warning: Failed to load {audio_path} efficiently: {e}")
            audio, sr = load_audio(str(audio_path), self.sample_rate, mono=True)
        
        # Normalize
        audio = normalize_audio(audio)
        
        # Ensure consistent chunk size (only pad if too short)
        if audio.shape[-1] < self.chunk_size:
            padding = self.chunk_size - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        # Clean audio is the target
        clean_audio = audio.clone()
        
        # Create degraded version
        if self.add_artifacts:
            degraded_audio = simulate_vinyl_artifacts(audio, self.sample_rate)
        else:
            degraded_audio = audio
        
        # Apply additional transforms if specified
        if self.transform:
            clean_audio = self.transform(clean_audio)
            degraded_audio = self.transform(degraded_audio)
        
        return degraded_audio, clean_audio


class StereoDataset(Dataset):
    """
    Dataset for stereo separation training
    Uses stereo audio and creates mono versions as input
    """
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 22050,
        chunk_duration: float = 2.0,
        transform: Optional[callable] = None
    ):
        """
        Args:
            data_dir: Directory containing stereo audio files
            sample_rate: Target sample rate
            chunk_duration: Duration of audio chunks in seconds
            transform: Optional transform
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.transform = transform
        
        # Find all audio files
        self.audio_files = []
        for ext in ['*.wav', '*.flac']:
            self.audio_files.extend(self.data_dir.glob(f"**/{ext}"))
        
        print(f"Found {len(self.audio_files)} audio files for stereo training")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mono_audio, stereo_audio) pair
        """
        audio_path = self.audio_files[idx]
        
        # Get file info first to check duration
        import soundfile as sf
        try:
            file_info = sf.info(str(audio_path))
            total_frames = file_info.frames
            
            # If file is longer than chunk, load only a random chunk
            if total_frames > self.chunk_size:
                # Pick random start position
                max_start = total_frames - self.chunk_size
                start_frame = np.random.randint(0, max_start + 1)
                
                # Load only the chunk we need (MUCH more memory efficient!)
                audio_data, sr = sf.read(
                    str(audio_path),
                    start=start_frame,
                    frames=self.chunk_size,
                    dtype='float32',
                    always_2d=True
                )
                audio = torch.from_numpy(audio_data.T)  # Transpose to (channels, samples)
            else:
                # File is short, load whole thing
                audio, sr = load_audio(str(audio_path), self.sample_rate, mono=False)
        except Exception as e:
            # Fallback to old method if something fails
            print(f"Warning: Failed to load {audio_path} efficiently: {e}")
            audio, sr = load_audio(str(audio_path), self.sample_rate, mono=False)
        
        # Ensure we have stereo
        if audio.shape[0] == 1:
            # Duplicate mono to stereo
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            # Take first 2 channels
            audio = audio[:2, :]
        
        # Normalize
        audio = normalize_audio(audio)
        
        # Ensure consistent chunk size (pad if too short)
        if audio.shape[-1] < self.chunk_size:
            padding = self.chunk_size - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        elif audio.shape[-1] > self.chunk_size:
            # Crop if somehow too long (shouldn't happen with new method)
            audio = audio[..., :self.chunk_size]
        
        # Stereo audio is the target
        stereo_audio = audio
        
        # Create mono version as input
        mono_audio = torch.mean(audio, dim=0, keepdim=True)
        
        if self.transform:
            mono_audio = self.transform(mono_audio)
            stereo_audio = self.transform(stereo_audio)
        
        return mono_audio, stereo_audio


def prepare_dataset(data_dir: str, batch_size: int = 16, num_workers: int = 4) -> DataLoader:
    """
    Prepare a DataLoader for training
    
    Args:
        data_dir: Directory containing audio files
        batch_size: Batch size
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    dataset = AudioRestorationDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


if __name__ == "__main__":
    # Test dataset
    print("Testing dataset creation...")
    # Note: This will fail without actual audio files
    # dataset = AudioRestorationDataset("data/raw")
    # print(f"Dataset size: {len(dataset)}")
