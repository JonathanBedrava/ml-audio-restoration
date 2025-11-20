"""
Analyze impulse characteristics from real 78rpm recordings
to improve synthetic artifact generation for training
"""

import torch
import numpy as np
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from .audio_processing import load_audio


def detect_impulses_analytical(
    audio: torch.Tensor,
    sample_rate: int,
    threshold_percentile: float = 99.5
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Detect and characterize impulses (crackles/pops) in audio
    
    Args:
        audio: Audio tensor (channels, samples)
        sample_rate: Sample rate
        threshold_percentile: Percentile for impulse detection threshold
    
    Returns:
        Tuple of (impulse_locations, impulse_amplitudes, statistics)
    """
    audio_np = audio.cpu().numpy()
    
    # Work with first channel if stereo
    if audio_np.shape[0] > 1:
        audio_np = audio_np[0]
    else:
        audio_np = audio_np[0]
    
    # Compute 1st and 2nd derivatives
    first_derivative = np.diff(audio_np, prepend=audio_np[0])
    second_derivative = np.diff(first_derivative, prepend=first_derivative[0])
    
    # Impulses show up as spikes in 2nd derivative
    abs_second_deriv = np.abs(second_derivative)
    
    # Adaptive threshold based on signal characteristics
    threshold = np.percentile(abs_second_deriv, threshold_percentile)
    
    # Find peaks above threshold
    peaks, properties = signal.find_peaks(
        abs_second_deriv,
        height=threshold,
        distance=int(sample_rate * 0.001)  # Min 1ms between impulses
    )
    
    # Get amplitudes
    amplitudes = abs_second_deriv[peaks]
    
    # Calculate statistics
    stats = {
        'num_impulses': len(peaks),
        'impulses_per_second': len(peaks) / (len(audio_np) / sample_rate),
        'mean_amplitude': np.mean(amplitudes) if len(amplitudes) > 0 else 0,
        'median_amplitude': np.median(amplitudes) if len(amplitudes) > 0 else 0,
        'max_amplitude': np.max(amplitudes) if len(amplitudes) > 0 else 0,
        'std_amplitude': np.std(amplitudes) if len(amplitudes) > 0 else 0,
        'threshold_used': threshold,
    }
    
    # Analyze inter-impulse intervals
    if len(peaks) > 1:
        intervals = np.diff(peaks) / sample_rate
        stats['mean_interval'] = np.mean(intervals)
        stats['median_interval'] = np.median(intervals)
        stats['min_interval'] = np.min(intervals)
    
    return peaks, amplitudes, stats


def analyze_frequency_content(
    audio: torch.Tensor,
    sample_rate: int,
    impulse_locations: np.ndarray,
    window_size: int = 512
) -> Dict:
    """
    Analyze frequency content of impulses vs background
    
    Args:
        audio: Audio tensor
        sample_rate: Sample rate
        impulse_locations: Sample indices of detected impulses
        window_size: FFT window size
    
    Returns:
        Dictionary with frequency analysis
    """
    audio_np = audio.cpu().numpy()
    if audio_np.shape[0] > 1:
        audio_np = audio_np[0]
    else:
        audio_np = audio_np[0]
    
    # Extract windows around impulses
    half_window = window_size // 2
    impulse_windows = []
    background_windows = []
    
    for loc in impulse_locations:
        if half_window < loc < len(audio_np) - half_window:
            impulse_windows.append(audio_np[loc - half_window:loc + half_window])
    
    # Sample background regions (far from impulses)
    safe_distance = int(sample_rate * 0.01)  # 10ms safety margin
    for _ in range(len(impulse_windows)):
        while True:
            start = np.random.randint(half_window, len(audio_np) - half_window)
            # Check if far enough from any impulse
            if np.all(np.abs(impulse_locations - start) > safe_distance):
                background_windows.append(audio_np[start - half_window:start + half_window])
                break
    
    # Compute average spectra
    if impulse_windows:
        impulse_fft = np.mean([np.abs(np.fft.rfft(w)) for w in impulse_windows], axis=0)
        background_fft = np.mean([np.abs(np.fft.rfft(w)) for w in background_windows], axis=0)
        
        freqs = np.fft.rfftfreq(window_size, 1/sample_rate)
        
        # Find dominant frequencies
        impulse_energy_ratio = impulse_fft / (background_fft + 1e-8)
        
        return {
            'freqs': freqs,
            'impulse_spectrum': impulse_fft,
            'background_spectrum': background_fft,
            'energy_ratio': impulse_energy_ratio,
            'high_freq_emphasis': np.mean(impulse_energy_ratio[freqs > 2000]),
            'mid_freq_emphasis': np.mean(impulse_energy_ratio[(freqs > 500) & (freqs < 2000)]),
        }
    
    return {}


def analyze_78rpm_recording(
    audio_path: str,
    sample_rate: int = 22050,
    plot: bool = True
) -> Dict:
    """
    Comprehensive analysis of a 78rpm recording
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        plot: Whether to generate visualization plots
    
    Returns:
        Dictionary with all analysis results
    """
    print(f"\nAnalyzing: {audio_path}")
    
    # Load audio
    audio, sr = load_audio(audio_path, sample_rate, mono=True)
    duration = audio.shape[-1] / sample_rate
    print(f"Duration: {duration:.2f} seconds")
    
    # Detect impulses
    peaks, amplitudes, stats = detect_impulses_analytical(audio, sample_rate)
    
    print(f"\nImpulse Statistics:")
    print(f"  Total impulses detected: {stats['num_impulses']}")
    print(f"  Impulses per second: {stats['impulses_per_second']:.2f}")
    print(f"  Mean amplitude: {stats['mean_amplitude']:.6f}")
    print(f"  Max amplitude: {stats['max_amplitude']:.6f}")
    if 'mean_interval' in stats:
        print(f"  Mean interval: {stats['mean_interval']:.3f} seconds")
        print(f"  Min interval: {stats['min_interval']:.3f} seconds")
    
    # Frequency analysis
    freq_analysis = analyze_frequency_content(audio, sample_rate, peaks)
    if freq_analysis:
        print(f"\nFrequency Analysis:")
        print(f"  High-freq emphasis (>2kHz): {freq_analysis['high_freq_emphasis']:.2f}x")
        print(f"  Mid-freq emphasis (0.5-2kHz): {freq_analysis['mid_freq_emphasis']:.2f}x")
    
    # Generate plots
    if plot and len(peaks) > 0:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Waveform with impulse markers
        ax = axes[0]
        audio_np = audio.cpu().numpy()[0]
        time = np.arange(len(audio_np)) / sample_rate
        ax.plot(time, audio_np, alpha=0.7, linewidth=0.5)
        ax.scatter(peaks / sample_rate, audio_np[peaks], color='red', s=10, alpha=0.5, label='Detected impulses')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Waveform with {len(peaks)} detected impulses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Amplitude distribution
        ax = axes[1]
        ax.hist(amplitudes, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(stats['mean_amplitude'], color='red', linestyle='--', label='Mean')
        ax.axvline(stats['median_amplitude'], color='green', linestyle='--', label='Median')
        ax.set_xlabel('Impulse Amplitude (2nd derivative)')
        ax.set_ylabel('Count')
        ax.set_title('Impulse Amplitude Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Frequency comparison
        if freq_analysis:
            ax = axes[2]
            freqs = freq_analysis['freqs']
            ax.semilogy(freqs, freq_analysis['impulse_spectrum'], label='Impulse', alpha=0.7)
            ax.semilogy(freqs, freq_analysis['background_spectrum'], label='Background', alpha=0.7)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude (log scale)')
            ax.set_title('Frequency Content: Impulses vs Background')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, sample_rate // 2)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(audio_path).parent / f"{Path(audio_path).stem}_impulse_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
        plt.close()
    
    return {
        'audio_path': audio_path,
        'duration': duration,
        'impulse_stats': stats,
        'frequency_analysis': freq_analysis,
        'peaks': peaks,
        'amplitudes': amplitudes,
    }


def compare_synthetic_vs_real(
    real_audio_path: str,
    clean_audio: torch.Tensor,
    sample_rate: int = 22050
) -> Dict:
    """
    Compare synthetic artifact generation with real 78rpm characteristics
    
    Args:
        real_audio_path: Path to real 78rpm recording
        clean_audio: Clean audio to add synthetic artifacts to
        sample_rate: Sample rate
    
    Returns:
        Comparison statistics
    """
    from .audio_processing import simulate_vinyl_artifacts
    
    # Analyze real recording
    real_results = analyze_78rpm_recording(real_audio_path, sample_rate, plot=False)
    
    # Generate synthetic version
    synthetic = simulate_vinyl_artifacts(clean_audio, sample_rate)
    
    # Analyze synthetic
    synth_peaks, synth_amps, synth_stats = detect_impulses_analytical(synthetic, sample_rate)
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON: Real vs Synthetic")
    print("="*60)
    print(f"\nImpulses per second:")
    print(f"  Real:      {real_results['impulse_stats']['impulses_per_second']:.2f}")
    print(f"  Synthetic: {synth_stats['impulses_per_second']:.2f}")
    
    print(f"\nMean amplitude:")
    print(f"  Real:      {real_results['impulse_stats']['mean_amplitude']:.6f}")
    print(f"  Synthetic: {synth_stats['mean_amplitude']:.6f}")
    
    print(f"\nMax amplitude:")
    print(f"  Real:      {real_results['impulse_stats']['max_amplitude']:.6f}")
    print(f"  Synthetic: {synth_stats['max_amplitude']:.6f}")
    
    return {
        'real': real_results,
        'synthetic': synth_stats,
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.analyze_impulses <audio_file_path>")
        print("\nExample: python -m src.utils.analyze_impulses test_audio/opera/old_recording.wav")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    if not Path(audio_path).exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)
    
    # Run analysis
    results = analyze_78rpm_recording(audio_path, sample_rate=22050, plot=True)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
