"""
UNet-based separation model with CLAP text conditioning.
Implements the core trainable model for text-guided audio separation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for conditioning."""
    def __init__(self, condition_dim: int, num_channels: int):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, num_channels)
        self.beta = nn.Linear(condition_dim, num_channels)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.
        Args:
            x: Feature map [B, C, H, W]
            condition: Condition vector [B, D]
        Returns:
            Modulated feature map [B, C, H, W]
        """
        gamma = self.gamma(condition).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta(condition).unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        return gamma * x + beta


class ConvBlock(nn.Module):
    """Convolutional block with optional FiLM conditioning."""
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.film = FiLMLayer(condition_dim, out_channels) if condition_dim else None
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        if self.film is not None and condition is not None:
            x = self.film(x, condition)
        
        return x


class TextConditionedUNet(nn.Module):
    """
    UNet model for predicting soft masks with CLAP text conditioning.
    
    Architecture:
    - Encoder: 4 downsampling blocks (64 -> 128 -> 256 -> 512 channels)
    - Bottleneck: 512 channels with FiLM conditioning
    - Decoder: 4 upsampling blocks with skip connections
    - Output: Soft mask in [0, 1] range
    """
    
    def __init__(self, clap_dim: int = 1024):
        super().__init__()
        
        # Input: 1 channel (magnitude spectrogram)
        self.encoder1 = ConvBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck with FiLM conditioning
        self.bottleneck = ConvBlock(512, 512, condition_dim=clap_dim)
        
        # Decoder with skip connections
        self.upconv4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.decoder4 = ConvBlock(512 + 512, 256, condition_dim=clap_dim)
        
        self.upconv3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.decoder3 = ConvBlock(256 + 256, 128, condition_dim=clap_dim)
        
        self.upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.decoder2 = ConvBlock(128 + 128, 64, condition_dim=clap_dim)
        
        self.upconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.decoder1 = ConvBlock(64 + 64, 64)
        
        # Output: 1 channel (soft mask)
        self.output = nn.Conv2d(64, 1, 1)
    
    def forward(self, mag_spec: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            mag_spec: Magnitude spectrogram [B, 1, F, T]
            text_emb: CLAP text embedding [B, 1024]
            
        Returns:
            Predicted soft mask [B, 1, F, T] in range [0, 1]
        """
        # Debug info
        print(f"Input mag_spec shape: {mag_spec.shape}")
        print(f"Input text_emb shape: {text_emb.shape}")
        
        # Ensure 4D input
        if mag_spec.dim() > 4:
            mag_spec = mag_spec.squeeze(1)  # Remove extra dimension if present
            
        # Ensure text embedding is 2D [B, 1024]
        if text_emb.dim() > 2:
            text_emb = text_emb.squeeze(1)
        
        # Encoder
        enc1 = self.encoder1(mag_spec)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        print(f"Encoder output shape: {enc4.shape}")
        
        # Bottleneck with text conditioning
        bottleneck = self.bottleneck(self.pool4(enc4), text_emb)
        
        # Decoder with skip connections and text conditioning
        dec4 = self.upconv4(bottleneck)
        # Match dimensions if needed
        if dec4.shape != enc4.shape:
            dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4, text_emb)
        
        dec3 = self.upconv3(dec4)
        if dec3.shape != enc3.shape:
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3, text_emb)
        
        dec2 = self.upconv2(dec3)
        if dec2.shape != enc2.shape:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2, text_emb)
        
        dec1 = self.upconv1(dec2)
        if dec1.shape != enc1.shape:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # Output mask with sigmoid activation for [0, 1] range
        mask = torch.sigmoid(self.output(dec1))
        
        return mask


class TextAgnosticUNet(nn.Module):
    """Baseline UNet without text conditioning for comparison."""
    
    def __init__(self):
        super().__init__()
        
        # Same architecture as TextConditionedUNet but without FiLM layers
        self.encoder1 = ConvBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = ConvBlock(512, 512)
        
        self.upconv4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.decoder4 = ConvBlock(512 + 512, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.decoder3 = ConvBlock(256 + 256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.decoder2 = ConvBlock(128 + 128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.decoder1 = ConvBlock(64 + 64, 64)
        
        self.output = nn.Conv2d(64, 1, 1)
    
    def forward(self, mag_spec: torch.Tensor) -> torch.Tensor:
        """Forward pass without text conditioning."""
        enc1 = self.encoder1(mag_spec)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        if dec4.shape != enc4.shape:
            dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        if dec3.shape != enc3.shape:
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        if dec2.shape != enc2.shape:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        if dec1.shape != enc1.shape:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        mask = torch.sigmoid(self.output(dec1))
        
        return mask


def compute_irm(target_mag: torch.Tensor, interferer_mag: torch.Tensor) -> torch.Tensor:
    """
    Compute Ideal Ratio Mask (IRM).
    
    Args:
        target_mag: Target magnitude spectrogram [B, F, T]
        interferer_mag: Interferer magnitude spectrogram [B, F, T]
        
    Returns:
        IRM [B, F, T] in range [0, 1]
    """
    denominator = target_mag + interferer_mag + 1e-8
    irm = target_mag / denominator
    return irm.clamp(0, 1)


if __name__ == "__main__":
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy input
    batch_size = 2
    freq_bins = 513  # n_fft // 2 + 1 for 1024 FFT
    time_frames = 100
    
    mag_spec = torch.randn(batch_size, 1, freq_bins, time_frames).to(device)
    text_emb = torch.randn(batch_size, 512).to(device)
    
    # Test text-conditioned model
    model = TextConditionedUNet(clap_dim=512).to(device)
    mask = model(mag_spec, text_emb)
    
    print(f"Input shape: {mag_spec.shape}")
    print(f"Output mask shape: {mask.shape}")
    print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test baseline model
    baseline = TextAgnosticUNet().to(device)
    mask_baseline = baseline(mag_spec)
    print(f"\nBaseline mask shape: {mask_baseline.shape}")
    print(f"Baseline parameters: {sum(p.numel() for p in baseline.parameters()):,}")
