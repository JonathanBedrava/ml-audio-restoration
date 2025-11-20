"""
Mixed dataset that combines:
1. Clean audio + synthetic artifacts (supervised)
2. Real degraded audio (semi-supervised/self-supervised)
3. Contrastive learning between synthetic and real
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from typing import Optional, Tuple
from .audio_processing import load_audio, normalize_audio, simulate_vinyl_artifacts


class MixedRestorationDataset(Dataset):
    """
    Advanced dataset that leverages both clean audio and real degraded recordings
    
    Training strategies:
    1. Supervised: clean + synthetic artifacts → clean
    2. Self-supervised: real degraded → denoised (consistency loss)
    3. Contrastive: synthetic vs real artifact discrimination
    4. Cycle consistency: clean → synthetic → denoised → clean
    """
    
    def __init__(
        self,
        clean_data_dir: str,
        degraded_data_dir: Optional[str] = None,
        sample_rate: int = 22050,
        chunk_duration: float = 2.0,
        synthetic_ratio: float = 0.7,  # 70% synthetic, 30% real
        use_contrastive: bool = True,
        use_cycle_consistency: bool = True,
        transform: Optional[callable] = None
    ):
        """
        Args:
            clean_data_dir: Directory with clean audio for synthetic artifacts
            degraded_data_dir: Directory with real degraded 78rpm recordings (optional)
            sample_rate: Target sample rate
            chunk_duration: Duration of audio chunks
            synthetic_ratio: Ratio of synthetic to real degraded samples
            use_contrastive: Enable contrastive learning between synthetic and real
            use_cycle_consistency: Enable cycle consistency loss
            transform: Optional transform
        """
        self.clean_dir = Path(clean_data_dir)
        self.degraded_dir = Path(degraded_data_dir) if degraded_data_dir else None
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.synthetic_ratio = synthetic_ratio
        self.use_contrastive = use_contrastive
        self.use_cycle_consistency = use_cycle_consistency
        self.transform = transform
        
        # Find clean audio files
        self.clean_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            self.clean_files.extend(self.clean_dir.glob(f"**/{ext}"))
        
        # Find real degraded files
        self.degraded_files = []
        if self.degraded_dir and self.degraded_dir.exists():
            for ext in ['*.wav', '*.mp3', '*.flac']:
                self.degraded_files.extend(self.degraded_dir.glob(f"**/{ext}"))
        
        print(f"Found {len(self.clean_files)} clean files")
        print(f"Found {len(self.degraded_files)} real degraded files")
        
        # Calculate split
        total_samples = len(self.clean_files)
        if len(self.degraded_files) > 0:
            # Mix synthetic and real
            self.num_synthetic = int(total_samples * synthetic_ratio)
            self.num_real = total_samples - self.num_synthetic
            print(f"Training mix: {self.num_synthetic} synthetic + {self.num_real} real samples")
        else:
            # All synthetic
            self.num_synthetic = total_samples
            self.num_real = 0
            print(f"Training with synthetic artifacts only ({self.num_synthetic} samples)")
    
    def __len__(self):
        return len(self.clean_files)
    
    def _load_and_chunk(self, file_path: Path, mono: bool = True) -> torch.Tensor:
        """Load audio and return random chunk"""
        audio, sr = load_audio(str(file_path), self.sample_rate, mono=mono)
        audio = normalize_audio(audio)
        
        # Get random chunk
        if audio.shape[-1] < self.chunk_size:
            padding = self.chunk_size - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        else:
            start = np.random.randint(0, audio.shape[-1] - self.chunk_size + 1)
            audio = audio[..., start:start + self.chunk_size]
        
        return audio
    
    def __getitem__(self, idx) -> dict:
        """
        Returns a dictionary with multiple training signals:
        - 'input': degraded audio (synthetic or real)
        - 'target': clean audio (or None for real degraded)
        - 'is_synthetic': whether artifacts are synthetic
        - 'contrastive_pair': optional pair for contrastive learning
        """
        result = {}
        
        # Determine if this sample uses synthetic or real degradation
        use_synthetic = (
            len(self.degraded_files) == 0 or 
            idx < self.num_synthetic
        )
        
        if use_synthetic:
            # Load clean audio
            clean_audio = self._load_and_chunk(self.clean_files[idx % len(self.clean_files)])
            
            # Apply synthetic artifacts
            degraded_audio = simulate_vinyl_artifacts(clean_audio, self.sample_rate)
            
            result['input'] = degraded_audio
            result['target'] = clean_audio
            result['is_synthetic'] = True
            
            # For cycle consistency: we have ground truth
            if self.use_cycle_consistency:
                result['cycle_target'] = clean_audio
        
        else:
            # Use real degraded recording
            real_idx = (idx - self.num_synthetic) % len(self.degraded_files)
            degraded_audio = self._load_and_chunk(self.degraded_files[real_idx])
            
            result['input'] = degraded_audio
            result['target'] = None  # No ground truth for real degraded
            result['is_synthetic'] = False
        
        # Contrastive pair: provide both synthetic and real for discrimination
        if self.use_contrastive and len(self.degraded_files) > 0:
            if use_synthetic:
                # Pair with a real degraded sample
                real_idx = np.random.randint(0, len(self.degraded_files))
                real_audio = self._load_and_chunk(self.degraded_files[real_idx])
                result['contrastive_pair'] = real_audio
                result['contrastive_label'] = 0  # 0 = different type
            else:
                # Pair with synthetic from same clean source
                clean_idx = np.random.randint(0, len(self.clean_files))
                clean = self._load_and_chunk(self.clean_files[clean_idx])
                synthetic = simulate_vinyl_artifacts(clean, self.sample_rate)
                result['contrastive_pair'] = synthetic
                result['contrastive_label'] = 0  # 0 = different type
        
        if self.transform:
            result['input'] = self.transform(result['input'])
            if result['target'] is not None:
                result['target'] = self.transform(result['target'])
        
        return result


class AdaptiveArtifactDataset(Dataset):
    """
    Dataset that learns artifact characteristics from real degraded audio
    and applies similar artifacts to clean training data
    """
    
    def __init__(
        self,
        clean_data_dir: str,
        reference_degraded_dir: str,
        sample_rate: int = 22050,
        chunk_duration: float = 2.0,
        analyze_every: int = 100,  # Re-analyze real audio every N epochs
    ):
        """
        Args:
            clean_data_dir: Clean audio directory
            reference_degraded_dir: Real 78rpm recordings to learn from
            sample_rate: Sample rate
            chunk_duration: Chunk duration
            analyze_every: How often to re-analyze real recordings
        """
        self.clean_dir = Path(clean_data_dir)
        self.degraded_dir = Path(reference_degraded_dir)
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.analyze_counter = 0
        self.analyze_every = analyze_every
        
        # Find files
        self.clean_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            self.clean_files.extend(self.clean_dir.glob(f"**/{ext}"))
        
        self.degraded_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            self.degraded_files.extend(self.degraded_dir.glob(f"**/{ext}"))
        
        print(f"Adaptive dataset: {len(self.clean_files)} clean, {len(self.degraded_files)} reference")
        
        # Analyze real degraded audio to extract artifact parameters
        self.artifact_params = self._analyze_real_artifacts()
    
    def _analyze_real_artifacts(self) -> dict:
        """
        Analyze real degraded recordings to extract artifact characteristics
        Returns parameters to match in synthetic generation
        """
        from .analyze_impulses import detect_impulses_analytical
        
        print("Analyzing real degraded audio for artifact characteristics...")
        
        impulse_rates = []
        impulse_amplitudes = []
        noise_levels = []
        
        # Sample a few degraded files
        num_samples = min(5, len(self.degraded_files))
        sample_indices = np.random.choice(len(self.degraded_files), num_samples, replace=False)
        
        for idx in sample_indices:
            audio, sr = load_audio(str(self.degraded_files[idx]), self.sample_rate, mono=True)
            
            # Detect impulses
            peaks, amplitudes, stats = detect_impulses_analytical(audio, self.sample_rate)
            
            if stats['num_impulses'] > 0:
                impulse_rates.append(stats['impulses_per_second'])
                impulse_amplitudes.append(stats['max_amplitude'])
            
            # Estimate noise level from quiet regions
            audio_np = audio.cpu().numpy().flatten()
            # Use bottom 10% amplitude regions as "noise floor"
            threshold = np.percentile(np.abs(audio_np), 10)
            noise_samples = audio_np[np.abs(audio_np) < threshold]
            if len(noise_samples) > 0:
                noise_levels.append(np.std(noise_samples))
        
        # Calculate average characteristics
        params = {
            'impulse_rate': np.mean(impulse_rates) if impulse_rates else 10.0,
            'impulse_rate_std': np.std(impulse_rates) if len(impulse_rates) > 1 else 5.0,
            'impulse_amplitude_max': np.mean(impulse_amplitudes) if impulse_amplitudes else 0.5,
            'noise_level': np.mean(noise_levels) if noise_levels else 0.02,
            'noise_level_std': np.std(noise_levels) if len(noise_levels) > 1 else 0.01,
        }
        
        print(f"Learned artifact parameters:")
        print(f"  Impulse rate: {params['impulse_rate']:.2f} ± {params['impulse_rate_std']:.2f} per second")
        print(f"  Max amplitude: {params['impulse_amplitude_max']:.4f}")
        print(f"  Noise level: {params['noise_level']:.4f} ± {params['noise_level_std']:.4f}")
        
        return params
    
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (degraded, clean) pair with learned artifact characteristics"""
        
        # Periodically re-analyze real recordings (artifacts may vary)
        self.analyze_counter += 1
        if self.analyze_counter >= self.analyze_every * len(self):
            self.artifact_params = self._analyze_real_artifacts()
            self.analyze_counter = 0
        
        # Load clean audio
        audio_path = self.clean_files[idx]
        audio, sr = load_audio(str(audio_path), self.sample_rate, mono=True)
        audio = normalize_audio(audio)
        
        # Random chunk
        if audio.shape[-1] < self.chunk_size:
            padding = self.chunk_size - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        else:
            start = np.random.randint(0, audio.shape[-1] - self.chunk_size + 1)
            audio = audio[..., start:start + self.chunk_size]
        
        clean_audio = audio.clone()
        
        # Apply artifacts with learned parameters (add randomness)
        impulse_rate = np.random.normal(
            self.artifact_params['impulse_rate'],
            self.artifact_params['impulse_rate_std']
        )
        impulse_rate = np.clip(impulse_rate, 1.0, 50.0)  # Reasonable bounds
        
        noise_level = np.random.normal(
            self.artifact_params['noise_level'],
            self.artifact_params['noise_level_std']
        )
        noise_level = np.clip(noise_level, 0.005, 0.1)
        
        degraded_audio = simulate_vinyl_artifacts(
            audio,
            self.sample_rate,
            impulse_rate=impulse_rate,
            impulse_amplitude=(0.1, self.artifact_params['impulse_amplitude_max']),
            surface_noise_level=(noise_level * 0.5, noise_level * 1.5),
            crackle_level=(noise_level * 0.3, noise_level * 0.8),
        )
        
        return degraded_audio, clean_audio


if __name__ == "__main__":
    # Test mixed dataset
    print("Testing MixedRestorationDataset...")
    
    dataset = MixedRestorationDataset(
        clean_data_dir="data/raw",
        degraded_data_dir="test_audio/opera",
        synthetic_ratio=0.7,
        use_contrastive=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Input shape: {sample['input'].shape}")
    print(f"Is synthetic: {sample['is_synthetic']}")
