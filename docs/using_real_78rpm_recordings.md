# Using Real 78rpm Recordings for Training

## Overview

The denoiser model includes specialized impulse detection to remove crackles and pops from 78rpm recordings. While synthetic artifacts are generated during training, real 78rpm recordings are valuable for:

1. **Validating** that synthetic artifacts match real-world characteristics
2. **Tuning** artifact generation parameters
3. **Testing** the model on authentic degraded audio

## Workflow

### 1. Collect Real 78rpm Recordings

Place real degraded 78rpm recordings in a test directory:

```
test_audio/
├── opera/
│   ├── caruso_aria.wav
│   ├── vintage_soprano.mp3
│   └── ...
├── jazz/
│   ├── old_blues.wav
│   └── ...
└── classical/
    ├── beethoven_78.wav
    └── ...
```

### 2. Analyze Impulse Characteristics

Run the impulse analyzer on real recordings:

```bash
python -m src.utils.analyze_impulses test_audio/opera/caruso_aria.wav
```

This will:
- Detect all impulses (crackles/pops)
- Measure amplitude distribution
- Analyze frequency content
- Generate visualization plots
- Save analysis results

**Output Statistics:**
```
Impulse Statistics:
  Total impulses detected: 423
  Impulses per second: 15.23
  Mean amplitude: 0.003421
  Max amplitude: 0.087654
  Mean interval: 0.066 seconds

Frequency Analysis:
  High-freq emphasis (>2kHz): 3.45x
  Mid-freq emphasis (0.5-2kHz): 1.87x
```

### 3. Tune Synthetic Artifact Generation

Based on real recording analysis, adjust parameters in `audio_processing.py`:

```python
# Example: If real recording shows ~15 impulses/second
synthetic = simulate_vinyl_artifacts(
    clean_audio,
    sample_rate=22050,
    impulse_rate=15.0,           # Match real rate
    impulse_amplitude=(0.1, 0.5), # Match real amplitude range
    surface_noise_level=(0.02, 0.04),
    crackle_level=(0.01, 0.025)
)
```

### 4. Compare Synthetic vs Real

Run comparison analysis:

```python
from src.utils.analyze_impulses import compare_synthetic_vs_real

results = compare_synthetic_vs_real(
    real_audio_path='test_audio/opera/caruso_aria.wav',
    clean_audio=your_clean_audio_tensor,
    sample_rate=22050
)
```

**Output:**
```
COMPARISON: Real vs Synthetic
============================================================

Impulses per second:
  Real:      15.23
  Synthetic: 14.87

Mean amplitude:
  Real:      0.003421
  Synthetic: 0.003156

Max amplitude:
  Real:      0.087654
  Synthetic: 0.082341
```

### 5. Training Configuration

The `AudioRestorationDataset` automatically applies synthetic artifacts during training:

```python
# In train_denoiser.py
dataset = AudioRestorationDataset(
    data_dir='data/raw',
    sample_rate=22050,
    chunk_duration=2.0,
    add_artifacts=True  # Synthetic artifacts enabled
)
```

**Training Process:**
1. Load clean audio from `data/raw/`
2. Apply `simulate_vinyl_artifacts()` with tuned parameters
3. Train model to map degraded → clean
4. Impulse detection loss emphasizes crackle/pop removal

### 6. Test on Real Recordings

After training, test the denoiser on real 78rpm recordings:

```bash
python src/inference.py \
    --model models/checkpoints/denoiser/checkpoint_best.pth \
    --input test_audio/opera/caruso_aria.wav \
    --output outputs/denoised/caruso_aria_restored.wav
```

Compare:
- Original degraded recording
- Denoised output
- Listen for crackle/pop removal effectiveness
- Check for preserved transients (consonants, attacks)

## Key Parameters

### Impulse Detection (in Denoiser Model)

The dual detection system combines:

**1. Learned Detection (Neural)**
```python
transient_detector = nn.Sequential(
    Conv1d(1, 32, 5), LeakyReLU,
    Conv1d(32, 16, 3), LeakyReLU,
    Conv1d(16, 8, 3), LeakyReLU,
    Conv1d(8, 1, 1), Sigmoid  # Soft mask
)
```

**2. Analytical Detection (Signal Processing)**
```python
# 1st and 2nd derivatives
first_deriv = audio[:, :, 1:] - audio[:, :, :-1]
second_deriv = first_deriv[:, :, 1:] - first_deriv[:, :, :-1]

# Amplitude weighting
amplitude = audio.abs()

# Combined impulse mask
impulse_mask = (second_deriv.abs() + amplitude) / 2
```

**Suppression:**
```python
suppression_mask = 1.0 - combined_mask * 0.9  # Up to 90% reduction
denoised = denoised * suppression_mask
```

### Artifact Generation Parameters

**Default Settings (matched to typical 78rpm):**
- `impulse_rate=10.0`: 10 pops/second (range: 5-20)
- `impulse_amplitude=(0.1, 0.5)`: Moderate to loud clicks
- `surface_noise_level=(0.015, 0.03)`: Continuous hiss
- `crackle_level=(0.01, 0.02)`: High-frequency crackle

**Heavy Degradation:**
```python
simulate_vinyl_artifacts(
    audio, sr,
    impulse_rate=25.0,           # Very damaged record
    impulse_amplitude=(0.2, 0.8), # Louder pops
    surface_noise_level=(0.03, 0.05),
    crackle_level=(0.02, 0.04)
)
```

**Light Degradation:**
```python
simulate_vinyl_artifacts(
    audio, sr,
    impulse_rate=5.0,
    impulse_amplitude=(0.05, 0.3),
    surface_noise_level=(0.01, 0.02),
    crackle_level=(0.005, 0.015)
)
```

## Best Practices

### 1. Diverse Training Data

Use clean audio from various genres:
- Opera (vocals with orchestra)
- Jazz (percussion, brass)
- Classical (strings, piano)
- Speech

This ensures the model learns to:
- Distinguish impulses from legitimate transients
- Preserve cymbal crashes vs remove clicks
- Keep consonants (plosives) vs suppress pops

### 2. Artifact Diversity

Vary synthetic artifact parameters during training:
```python
# Random severity in dataset
impulse_rate = np.random.uniform(5, 20)
impulse_amp_max = np.random.uniform(0.3, 0.8)
```

### 3. Validation Strategy

Test on:
- Synthetic artifacts (in-distribution)
- Real 78rpm recordings (out-of-distribution)
- Edge cases: quiet passages, dense transients

### 4. Loss Weight Tuning

If model over-smooths or under-removes:

**Over-smoothing (too aggressive):**
```python
# In trainer.py
impulse_loss_weight = 0.2  # Reduce from 0.3
```

**Under-removing (too conservative):**
```python
# In trainer.py
impulse_loss_weight = 0.4  # Increase from 0.3

# Or in denoiser.py
suppression_mask = 1.0 - combined_mask * 0.95  # Up to 95% reduction
```

## Example: Complete Analysis Pipeline

```bash
# 1. Analyze real recording
python -m src.utils.analyze_impulses test_audio/opera/old_recording.wav

# 2. Review generated plot
# Check: test_audio/opera/old_recording_impulse_analysis.png

# 3. Note parameters from output
# Example: 18.5 impulses/sec, max amp 0.065

# 4. Update synthetic generation
# Edit src/utils/audio_processing.py
# Set impulse_rate=18.0, impulse_amplitude=(0.1, 0.6)

# 5. Train with tuned parameters
python src/training/train_denoiser.py

# 6. Test on real recording
python src/inference.py \
    --model models/checkpoints/denoiser/checkpoint_best.pth \
    --input test_audio/opera/old_recording.wav \
    --output outputs/restored.wav

# 7. Listen and compare
# Check for:
#   - Removed crackles/pops ✓
#   - Preserved vocal sibilants ✓
#   - Preserved orchestra attacks ✓
#   - Natural sound (no over-processing) ✓
```

## Troubleshooting

### False Positives (Removing Legitimate Transients)

**Symptom:** Drum hits, consonants sound muffled

**Solutions:**
1. Reduce impulse loss weight: `impulse_loss_weight = 0.2`
2. Reduce suppression: `suppression_mask = 1.0 - mask * 0.8`
3. Add more percussive content to training data
4. Train longer (model learns discrimination)

### False Negatives (Not Removing All Clicks)

**Symptom:** Some crackles/pops remain audible

**Solutions:**
1. Increase impulse loss weight: `impulse_loss_weight = 0.4`
2. Increase suppression: `suppression_mask = 1.0 - mask * 0.95`
3. Increase synthetic impulse rate during training
4. Add more varied impulse amplitudes

### Artifacts Sound "Unnatural"

**Symptom:** Synthetic artifacts don't match real recordings

**Solutions:**
1. Analyze multiple real 78rpm recordings
2. Calculate average impulse characteristics
3. Tune generation parameters to match
4. Add resonance and ringing to synthetic impulses
5. Use asymmetric impulse polarities

## Metrics

Track these during validation:

1. **SNR (Signal-to-Noise Ratio):** Overall noise reduction
2. **Impulse Removal Rate:** % of detected impulses removed
3. **Transient Preservation:** Correlation of drum/piano attacks
4. **Spectral Flatness:** Avoid over-smoothing high frequencies
5. **PESQ/POLQA:** Perceptual quality scores (if available)

## References

- Dual impulse detection: `src/models/denoiser.py`
- Artifact generation: `src/utils/audio_processing.py`
- Impulse analysis: `src/utils/analyze_impulses.py`
- Training pipeline: `src/training/train_denoiser.py`
- Loss functions: `src/training/trainer.py` (impulse_loss method)
