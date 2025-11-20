"""
Enhanced trainer with semi-supervised learning for real degraded audio
Supports multiple training modes:
1. Supervised: clean + synthetic → clean (standard)
2. Semi-supervised: real degraded → consistency regularization
3. Contrastive: discriminate synthetic vs real artifacts
4. Cycle consistency: clean → degraded → restored → clean
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class SemiSupervisedLoss(nn.Module):
    """
    Combined loss for training with both clean+synthetic and real degraded data
    """
    
    def __init__(
        self,
        supervised_weight: float = 1.0,
        consistency_weight: float = 0.3,
        contrastive_weight: float = 0.1,
        cycle_weight: float = 0.2,
    ):
        """
        Args:
            supervised_weight: Weight for supervised loss (clean target available)
            consistency_weight: Weight for consistency on real degraded (no target)
            contrastive_weight: Weight for contrastive learning
            cycle_weight: Weight for cycle consistency
        """
        super().__init__()
        self.supervised_weight = supervised_weight
        self.consistency_weight = consistency_weight
        self.contrastive_weight = contrastive_weight
        self.cycle_weight = cycle_weight
    
    def supervised_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Standard MSE loss when ground truth is available"""
        return F.mse_loss(output, target)
    
    def consistency_loss(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """
        Consistency regularization for real degraded audio (no ground truth)
        
        Enforces:
        1. Output should be smoother than input (fewer impulses)
        2. Output should preserve spectral envelope
        3. Output energy should be similar to input
        """
        # Temporal smoothness: penalize rapid changes (impulses)
        output_diff = output[:, :, 1:] - output[:, :, :-1]
        input_diff = input[:, :, 1:] - input[:, :, :-1]
        
        # Output should have smaller derivatives than input (smoother)
        smoothness_loss = F.relu(output_diff.abs().mean() - input_diff.abs().mean() * 0.5)
        
        # Energy preservation: total energy should be similar
        output_energy = (output ** 2).sum(dim=-1)
        input_energy = (input ** 2).sum(dim=-1)
        energy_loss = F.mse_loss(output_energy, input_energy)
        
        # Spectral envelope preservation (low-frequency structure)
        # Use larger FFT window to capture envelope, not fine details
        n_fft = min(2048, input.shape[-1])
        
        output_fft = torch.fft.rfft(output, n=n_fft, dim=-1)
        input_fft = torch.fft.rfft(input, n=n_fft, dim=-1)
        
        # Compare magnitude spectra (preserve overall frequency content)
        output_mag = output_fft.abs()
        input_mag = input_fft.abs()
        
        # Use log scale for better perceptual weighting
        spectral_loss = F.l1_loss(
            torch.log(output_mag + 1e-8),
            torch.log(input_mag + 1e-8)
        )
        
        return smoothness_loss * 0.3 + energy_loss * 0.2 + spectral_loss * 0.5
    
    def contrastive_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive loss to learn discriminative features
        
        Args:
            features_a: Features from sample A
            features_b: Features from sample B
            label: 1 if same type (both synthetic or both real), 0 if different
        
        Helps model learn what makes synthetic different from real artifacts
        """
        # Cosine similarity
        similarity = F.cosine_similarity(features_a, features_b, dim=-1)
        
        # Pull together if same type, push apart if different
        target_similarity = label.float()  # 1 for same, 0 for different
        
        loss = F.mse_loss(similarity, target_similarity)
        return loss
    
    def cycle_consistency_loss(
        self,
        restored: torch.Tensor,
        original_clean: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """
        Cycle consistency: clean → add_artifacts → denoise → should equal clean
        
        This ensures the model can fully remove the artifacts it was trained on
        """
        # Re-apply artifacts to the restored audio
        from src.utils.audio_processing import simulate_vinyl_artifacts
        
        # This happens in training mode, so no gradient through artifact simulation
        with torch.no_grad():
            re_degraded = []
            for i in range(restored.shape[0]):
                sample = simulate_vinyl_artifacts(
                    restored[i:i+1],
                    22050  # Assuming 22050 Hz, could be passed as parameter
                )
                re_degraded.append(sample)
            re_degraded = torch.cat(re_degraded, dim=0)
        
        # Denoise again
        re_restored = model(re_degraded)
        
        # Should match the first restoration
        cycle_loss = F.mse_loss(re_restored, restored)
        
        # Also should still be close to original clean
        clean_loss = F.mse_loss(restored, original_clean)
        
        return cycle_loss * 0.5 + clean_loss * 0.5
    
    def forward(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        is_synthetic: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            output: Model output (denoised)
            input: Model input (degraded)
            target: Ground truth (clean), None for real degraded samples
            is_synthetic: Boolean tensor indicating synthetic samples
            model: The model itself (for cycle consistency)
        
        Returns:
            Dictionary with total loss and component losses
        """
        losses = {}
        total_loss = 0.0
        
        # Supervised loss (for samples with ground truth)
        if target is not None and is_synthetic is not None:
            # Only compute for synthetic samples (have ground truth)
            mask = is_synthetic.bool()
            if mask.any():
                supervised = self.supervised_loss(
                    output[mask],
                    target[mask]
                )
                losses['supervised'] = supervised
                total_loss += supervised * self.supervised_weight
        elif target is not None:
            # All samples have ground truth
            supervised = self.supervised_loss(output, target)
            losses['supervised'] = supervised
            total_loss += supervised * self.supervised_weight
        
        # Consistency loss (for real degraded samples without ground truth)
        if is_synthetic is not None and self.consistency_weight > 0:
            mask = ~is_synthetic.bool()
            if mask.any():
                consistency = self.consistency_loss(
                    output[mask],
                    input[mask]
                )
                losses['consistency'] = consistency
                total_loss += consistency * self.consistency_weight
        
        # Contrastive loss (if contrastive pairs provided)
        if 'contrastive_pair' in kwargs and self.contrastive_weight > 0:
            # Extract features from intermediate layer
            # This would require model to return features, or use a feature extractor
            pass  # TODO: Implement if needed
        
        # Cycle consistency (if enabled and model provided)
        if target is not None and model is not None and self.cycle_weight > 0:
            if is_synthetic is not None:
                mask = is_synthetic.bool()
                if mask.any():
                    cycle = self.cycle_consistency_loss(
                        output[mask],
                        target[mask],
                        model
                    )
                    losses['cycle'] = cycle
                    total_loss += cycle * self.cycle_weight
        
        losses['total'] = total_loss
        return losses


def train_with_mixed_data(
    model: nn.Module,
    mixed_dataloader,
    optimizer: torch.optim.Optimizer,
    device: str = 'cuda',
    use_amp: bool = True
) -> Dict[str, float]:
    """
    Training loop for mixed dataset (synthetic + real degraded)
    
    Args:
        model: The denoiser model
        mixed_dataloader: DataLoader with MixedRestorationDataset
        optimizer: Optimizer
        device: Device to train on
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        Dictionary with average losses
    """
    model.train()
    
    loss_fn = SemiSupervisedLoss(
        supervised_weight=1.0,
        consistency_weight=0.3,
        contrastive_weight=0.0,  # Disabled for now
        cycle_weight=0.2
    )
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    total_losses = {
        'total': 0.0,
        'supervised': 0.0,
        'consistency': 0.0,
        'cycle': 0.0,
        'count_synthetic': 0,
        'count_real': 0,
    }
    
    for batch_idx, batch in enumerate(mixed_dataloader):
        # Batch is a dictionary with 'input', 'target', 'is_synthetic', etc.
        input_audio = batch['input'].to(device)
        target_audio = batch.get('target')  # May be None for some samples
        is_synthetic = batch['is_synthetic'].to(device)
        
        if target_audio is not None:
            target_audio = target_audio.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(input_audio)
                losses = loss_fn(
                    output=output,
                    input=input_audio,
                    target=target_audio,
                    is_synthetic=is_synthetic,
                    model=model
                )
                loss = losses['total']
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(input_audio)
            losses = loss_fn(
                output=output,
                input=input_audio,
                target=target_audio,
                is_synthetic=is_synthetic,
                model=model
            )
            loss = losses['total']
            
            loss.backward()
            optimizer.step()
        
        # Accumulate losses
        total_losses['total'] += loss.item()
        for key in ['supervised', 'consistency', 'cycle']:
            if key in losses:
                total_losses[key] += losses[key].item()
        
        total_losses['count_synthetic'] += is_synthetic.sum().item()
        total_losses['count_real'] += (~is_synthetic).sum().item()
    
    # Average losses
    num_batches = len(mixed_dataloader)
    for key in ['total', 'supervised', 'consistency', 'cycle']:
        total_losses[key] /= num_batches
    
    return total_losses


if __name__ == "__main__":
    print("Semi-supervised training utilities for mixed datasets")
    print("\nThis module enables training with:")
    print("  1. Clean audio + synthetic artifacts (supervised)")
    print("  2. Real degraded 78rpm recordings (semi-supervised)")
    print("  3. Consistency regularization for real data")
    print("  4. Cycle consistency for robustness")
