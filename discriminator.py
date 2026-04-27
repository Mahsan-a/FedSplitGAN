"""
discriminator.py
----------------
Site-local discriminator for the FedSplitGAN framework.

Each participating clinical site hosts one instance of this network. The
discriminator receives:
  - Real scalograms from its local patient population (never transmitted)
  - Synthetic scalograms broadcast from the central generator

It computes the adversarial loss and returns a gradient signal ∇D that flows
back through the communication layer to update the central generator. Raw
patient data never leaves the site.

Architecture: PatchGAN-style discriminator operating on (2, 96, 288) scalograms.
PatchGAN is preferred over a single fully-connected output because it:
  1. Produces richer gradient signal per forward pass (one prediction per patch)
  2. Is less prone to mode collapse when gradients are aggregated from multiple sites
  3. Naturally enforces local spectral coherence in the generated scalograms

Gradient penalty (WGAN-GP) is applied locally at each site before the gradient
is compressed and transmitted, which stabilises federated GAN training where
discriminator updates at different sites can diverge.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import Tuple


# ── Discriminator block ────────────────────────────────────────────────────
def _disc_block(in_ch: int, out_ch: int, stride: int = 2,
                use_sn: bool = True) -> nn.Module:
    conv = nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=False)
    if use_sn:
        conv = spectral_norm(conv)
    return nn.Sequential(conv, nn.LeakyReLU(0.2, inplace=True))


# ── PatchGAN Discriminator ─────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    PatchGAN discriminator for (2, 96, 288) dual-channel CGM scalograms.

    Input:  x ∈ R^{2 × 96 × 288}   (real or synthetic scalogram)
    Output: patch_scores ∈ R^{1 × H' × W'}  — one score per receptive field patch

    Stride pattern:  96×288  →  48×144  →  24×72  →  12×36  →  6×18  →  1×1
    """

    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            # No BatchNorm in first layer (standard PatchGAN practice)
            _disc_block(in_channels, 64,  stride=2, use_sn=True),  # 48×144
            _disc_block(64,          128, stride=2, use_sn=True),  # 24×72
            nn.BatchNorm2d(128),
            _disc_block(128,         256, stride=2, use_sn=True),  # 12×36
            nn.BatchNorm2d(256),
            _disc_block(256,         512, stride=2, use_sn=True),  # 6×18
            nn.BatchNorm2d(512),
            # Final patch prediction
            spectral_norm(nn.Conv2d(512, 1, 4, stride=1, padding=1)),  # 6×18 → 5×17
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2, 96, 288)
        Returns:
            scores: (batch, 1, H', W')  — raw logits, no sigmoid
                    (sigmoid applied in loss computation for numerical stability)
        """
        return self.net(x)


# ── Conditional Discriminator ──────────────────────────────────────────────
class ConditionalDiscriminator(nn.Module):
    """
    Conditional PatchGAN discriminator that additionally accepts a risk-group
    label, conditioning discrimination on patient subgroup membership.
    The label embedding is projected to a spatial map and concatenated with
    the input along the channel dimension before passing through the network.
    """

    def __init__(self, in_channels: int = 2, n_classes: int = 4,
                 embed_dim: int = 16, img_h: int = 96, img_w: int = 288):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        self.img_h = img_h
        self.img_w = img_w
        self.disc  = Discriminator(in_channels=in_channels + embed_dim)

    def forward(self, x: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      (batch, 2, 96, 288)
            labels: (batch,)  int64
        Returns:
            scores: (batch, 1, H', W')
        """
        emb = self.embed(labels)                             # (batch, embed_dim)
        # Tile embedding spatially to (batch, embed_dim, H, W)
        emb_map = emb[:, :, None, None].expand(
            -1, -1, self.img_h, self.img_w)
        x_cond = torch.cat([x, emb_map], dim=1)             # (batch, 2+embed, H, W)
        return self.disc(x_cond)


# ── Loss Functions ─────────────────────────────────────────────────────────
def discriminator_loss(real_scores: torch.Tensor,
                       fake_scores: torch.Tensor,
                       mode: str = "bce") -> torch.Tensor:
    """
    Args:
        real_scores: D(x_real) patch map
        fake_scores: D(x_syn)  patch map
        mode: "bce" (standard GAN) | "hinge" | "wgan"
    """
    if mode == "bce":
        ones  = torch.ones_like(real_scores)
        zeros = torch.zeros_like(fake_scores)
        return (nn.functional.binary_cross_entropy_with_logits(real_scores, ones) +
                nn.functional.binary_cross_entropy_with_logits(fake_scores, zeros))
    elif mode == "hinge":
        return (nn.functional.relu(1.0 - real_scores).mean() +
                nn.functional.relu(1.0 + fake_scores).mean())
    elif mode == "wgan":
        return -(real_scores.mean() - fake_scores.mean())
    else:
        raise ValueError(f"Unknown loss mode: {mode}")


def generator_loss(fake_scores: torch.Tensor, mode: str = "bce") -> torch.Tensor:
    if mode == "bce":
        return nn.functional.binary_cross_entropy_with_logits(
            fake_scores, torch.ones_like(fake_scores))
    elif mode == "hinge":
        return -fake_scores.mean()
    elif mode == "wgan":
        return -fake_scores.mean()
    else:
        raise ValueError(f"Unknown loss mode: {mode}")


def gradient_penalty(discriminator: nn.Module,
                     real: torch.Tensor,
                     fake: torch.Tensor,
                     device: torch.device,
                     lambda_gp: float = 10.0) -> torch.Tensor:
    """
    WGAN-GP gradient penalty, applied locally at each client site before
    the discriminator gradient is compressed and transmitted.

    This local application means the GP enforces a Lipschitz constraint on
    each site's discriminator independently — appropriate because each site
    has a distinct real data distribution.
    """
    batch = real.size(0)
    alpha = torch.rand(batch, 1, 1, 1, device=device).expand_as(real)
    interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    d_interp = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interp, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(batch, -1)
    penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = Discriminator().to(device)
    x = torch.randn(4, 2, 96, 288, device=device)
    scores = d(x)
    print(f"Discriminator output shape: {scores.shape}")  # (4, 1, H', W')
    print(f"Discriminator parameters  : {sum(p.numel() for p in d.parameters()):,}")
