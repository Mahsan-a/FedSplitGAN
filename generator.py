"""
generator.py
------------
Central generator for the FedSplitGAN framework.

Architecture: noise vector z ∈ R^{latent_dim}  →  synthetic CGM scalogram (2, 96, 288)

Design choices:
  - Transposed convolution upsampling — avoids checkerboard artifacts relative
    to nearest-neighbour upsampling + conv in the frequency dimension.
  - Separate projection heads for the two wavelet channels, concatenated at the
    final layer, so the generator can learn channel-specific spectral structure.
  - Spectral normalization on generator weights improves training stability in
    the federated setting where gradient signals arrive asynchronously from
    multiple discriminators.
  - Conditional variant (CGANGenerator) accepts a risk-group label embedding,
    enabling class-conditional generation to oversample rare-event patient
    scalogram patterns for downstream data augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ── Residual Block ─────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """Residual block with spectral norm and batch norm."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))


# ── Generator ──────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    Unconditional central generator.

    Input:  z ∈ R^{latent_dim}  (sampled from N(0,I))
    Output: x_syn ∈ R^{2 × 96 × 288}  — dual-channel CGM scalogram

    Upsampling path:
        z (latent_dim)
        → FC → reshape to (512, 6, 9)
        → ConvT 256  (6→12,   9→18)
        → ConvT 128  (12→24,  18→36)
        → ConvT  64  (24→48,  36→72)
        → ConvT  32  (48→96,  72→144)
        → Conv   2   (96×144 → 96×288 via stride-1 + padding)
        → Tanh

    The spatial target is (96, 288) — matching the CWT scalogram shape.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Project and reshape: z → (512, 6, 9)
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 512 * 6 * 9)),
            nn.ReLU(inplace=True)
        )

        # Upsampling blocks: (6,9) → (96,288) in four transposed-conv steps
        # plus a final conv to fix 144→288 in width
        self.up1 = self._up_block(512, 256)   # → (12, 18)
        self.res1 = ResBlock(256)

        self.up2 = self._up_block(256, 128)   # → (24, 36)
        self.res2 = ResBlock(128)

        self.up3 = self._up_block(128, 64)    # → (48, 72)
        self.res3 = ResBlock(64)

        self.up4 = self._up_block(64, 32)     # → (96, 144)
        self.res4 = ResBlock(32)

        # Widen from 144 → 288 with a final Conv
        self.final = nn.Sequential(
            spectral_norm(nn.Conv2d(32, 16, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1))),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Output: 2 channels (MexHat + Morlet)
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    @staticmethod
    def _up_block(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            x_syn: (batch, 2, 96, 288)  values in [-1, 1]
        """
        x = self.fc(z).view(-1, 512, 6, 9)
        x = self.res1(self.up1(x))
        x = self.res2(self.up2(x))
        x = self.res3(self.up3(x))
        x = self.res4(self.up4(x))
        return self.final(x)


# ── Conditional Generator ──────────────────────────────────────────────────
class CGANGenerator(nn.Module):
    """
    Conditional generator: accepts a risk-group label c ∈ {0,1,2,3}
    (rare / occasional / frequent / high-risk) alongside the noise vector z.

    The label is embedded into a dense vector and concatenated with z before
    the FC projection, enabling class-conditional synthetic scalogram generation.
    This is the recommended variant for rare-event data augmentation, where
    you want to oversample the rare (c=0) class specifically.
    """

    def __init__(self, latent_dim: int = 128, n_classes: int = 4,
                 embed_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.gen = Generator(latent_dim=latent_dim + embed_dim)

    def forward(self, z: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:      (batch, latent_dim)
            labels: (batch,)  int64 risk-group labels
        Returns:
            x_syn: (batch, 2, 96, 288)
        """
        emb = self.embedding(labels)          # (batch, embed_dim)
        z_cond = torch.cat([z, emb], dim=1)   # (batch, latent_dim + embed_dim)
        return self.gen(z_cond)


# ── Utility ────────────────────────────────────────────────────────────────
def sample_noise(batch_size: int, latent_dim: int,
                 device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, latent_dim, device=device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = Generator(latent_dim=128).to(device)
    z = sample_noise(4, 128, device)
    out = g(z)
    print(f"Generator output shape : {out.shape}")   # (4, 2, 96, 288)
    print(f"Generator parameters   : {count_parameters(g):,}")

    cg = CGANGenerator(latent_dim=128, n_classes=4, embed_dim=32).to(device)
    labels = torch.zeros(4, dtype=torch.long, device=device)  # all 'rare' class
    cout = cg(z, labels)
    print(f"cGAN generator output  : {cout.shape}")  # (4, 2, 96, 288)
