"""
Inference script for audio restoration
Applies denoising, super-resolution, and stereo separation to input audio
"""
import sys
sys.path.append('.')

import torch
import argparse
from pathlib import Path
from src.models.denoiser import AudioDenoiser
from src.models.stereo_separator import StereoSeparator
from src.models.super_resolution import AudioSuperResolution
from src.utils.audio_processing import load_audio, save_audio, normalize_audio


def restore_audio(
    input_path: str,
    output_path: str,
    denoiser_checkpoint: str = 'models/checkpoints/best_model.pth',
    super_res_checkpoint: str = 'models/checkpoints/super_resolution/best_model.pth',
    stereo_checkpoint: str = 'models/checkpoints/stereo/best_model.pth',
    sample_rate: int = 22050,
    enable_super_resolution: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Restore audio by applying denoising, super-resolution, and stereo separation
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save restored audio
        denoiser_checkpoint: Path to denoiser model checkpoint
        super_res_checkpoint: Path to super-resolution model checkpoint
        stereo_checkpoint: Path to stereo separator checkpoint
        sample_rate: Sample rate for processing
        enable_super_resolution: Whether to apply bandwidth extension
        device: Device to run inference on
    """
    print(f"Processing: {input_path}")
    print(f"Device: {device}")
    
    # Load audio
    print("Loading audio...")
    audio, sr = load_audio(input_path, sample_rate=sample_rate, mono=True)
    audio = normalize_audio(audio)
    audio = audio.to(device)
    
    # Load denoiser model
    print("Loading denoiser model...")
    denoiser = AudioDenoiser()
    checkpoint = torch.load(denoiser_checkpoint, map_location=device)
    denoiser.load_state_dict(checkpoint['model_state_dict'])
    denoiser = denoiser.to(device)
    denoiser.eval()
    
    # Apply denoising
    print("Applying denoising...")
    with torch.no_grad():
        denoised = denoiser(audio.unsqueeze(0))
        denoised = denoised.squeeze(0)
    
    # Apply super-resolution if enabled
    if enable_super_resolution:
        print("Loading super-resolution model...")
        super_res_model = AudioSuperResolution(upscale_factor=2)
        checkpoint = torch.load(super_res_checkpoint, map_location=device)
        super_res_model.load_state_dict(checkpoint['model_state_dict'])
        super_res_model = super_res_model.to(device)
        super_res_model.eval()
        
        print("Applying bandwidth extension (22.05kHz -> 44.1kHz)...")
        with torch.no_grad():
            enhanced = super_res_model(denoised.unsqueeze(0))
            enhanced = enhanced.squeeze(0)
        
        # Update sample rate for subsequent processing
        sample_rate = sample_rate * 2
        audio_to_stereo = enhanced
    else:
        audio_to_stereo = denoised
    
    # Load stereo separator model
    print("Loading stereo separator model...")
    stereo_model = StereoSeparator()
    checkpoint = torch.load(stereo_checkpoint, map_location=device)
    stereo_model.load_state_dict(checkpoint['model_state_dict'])
    stereo_model = stereo_model.to(device)
    stereo_model.eval()
    
    # Apply stereo separation
    print("Applying stereo separation...")
    with torch.no_grad():
        stereo = stereo_model(audio_to_stereo.unsqueeze(0))
        stereo = stereo.squeeze(0)
    
    # Normalize output
    stereo = normalize_audio(stereo)
    
    # Save result
    print(f"Saving to: {output_path}")
    save_audio(output_path, stereo.cpu(), sample_rate)
    
    print("Restoration complete!")
    if enable_super_resolution:
        print(f"Output sample rate: {sample_rate}Hz (bandwidth extended)")
    else:
        print(f"Output sample rate: {sample_rate}Hz")


def main():
    parser = argparse.ArgumentParser(description='Restore 78rpm record audio')
    parser.add_argument('input', type=str, help='Input audio file path')
    parser.add_argument('output', type=str, help='Output audio file path')
    parser.add_argument('--denoiser', type=str, 
                       default='models/checkpoints/best_model.pth',
                       help='Path to denoiser checkpoint')
    parser.add_argument('--super-res', type=str,
                       default='models/checkpoints/super_resolution/best_model.pth',
                       help='Path to super-resolution checkpoint')
    parser.add_argument('--stereo', type=str,
                       default='models/checkpoints/stereo/best_model.pth',
                       help='Path to stereo separator checkpoint')
    parser.add_argument('--sample-rate', type=int, default=22050,
                       help='Sample rate for processing')
    parser.add_argument('--no-super-res', action='store_true',
                       help='Disable bandwidth extension (super-resolution)')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    restore_audio(
        input_path=args.input,
        output_path=args.output,
        denoiser_checkpoint=args.denoiser,
        super_res_checkpoint=args.super_res,
        stereo_checkpoint=args.stereo,
        sample_rate=args.sample_rate,
        enable_super_resolution=not args.no_super_res,
        device=args.device
    )


if __name__ == "__main__":
    main()
