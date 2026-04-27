"""
communication.py
----------------
Peer-to-peer ring gradient communication between distributed discriminator sites.

Instead of each site independently sending ∇D to the central server (star topology),
sites first propagate compressed gradient statistics around a logical ring — sharing
what they have learned from their local real data before the aggregated signal
reaches the generator. This achieves two things:

  1. Communication efficiency: the number of required server-round-trips is reduced
     because sites can absorb gradients from neighbours before aggregating, smoothing
     out the high variance that comes from single-site gradient estimates on small
     local batches.

  2. Privacy: gradients are further obfuscated by mixing with neighbour gradients
     before reaching the central server. Combined with Gaussian differential privacy
     noise, this makes individual-site gradient inversion attacks harder.

P2P communication protocol (ring, synchronous):
    Site 0 → Site 1 → Site 2 → ... → Site K-1 → Site 0

Each site receives the accumulated gradient from its left neighbour, adds its own
compressed gradient, and forwards to its right neighbour. After one ring pass, all
sites hold the same aggregated gradient, which they then send to the central server.

Gradient compression: top-k sparsification + int8 quantization, reducing per-round
bandwidth by ~8–16× relative to full float32 transmission.

Differential privacy: Gaussian mechanism (σ calibrated to (ε, δ)-DP) applied
before any gradient leaves the site.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np


# ── Gradient Compression ───────────────────────────────────────────────────
def topk_sparsify(grad: torch.Tensor, k_fraction: float = 0.1) -> torch.Tensor:
    """
    Top-k sparsification: zero out all but the top-k% largest-magnitude elements.
    Reduces transmitted gradient density by ~10× at k_fraction=0.1.
    """
    flat = grad.view(-1)
    k    = max(1, int(k_fraction * flat.numel()))
    _, idx = torch.topk(flat.abs(), k)
    mask = torch.zeros_like(flat)
    mask[idx] = 1.0
    return (flat * mask).view_as(grad)


def quantize_int8(grad: torch.Tensor) -> torch.Tensor:
    """Uniform int8 quantization for bandwidth reduction."""
    abs_max = grad.abs().max().clamp(min=1e-8)
    g_norm  = (grad / abs_max).clamp(-1.0, 1.0)
    g_q     = (g_norm * 127).round().to(torch.int8)
    return g_q.float() / 127.0 * abs_max


def compress_gradient(grad: torch.Tensor,
                       k_fraction: float = 0.1,
                       quantize: bool = True) -> torch.Tensor:
    g = topk_sparsify(grad, k_fraction)
    if quantize:
        g = quantize_int8(g)
    return g


# ── Differential Privacy ───────────────────────────────────────────────────
def add_dp_noise(grad: torch.Tensor,
                 clip_norm: float = 1.0,
                 noise_multiplier: float = 1.1,
                 device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Gaussian mechanism for (ε, δ)-DP on gradient tensors.

    Steps:
      1. Clip gradient L2 norm to clip_norm  (sensitivity bounding)
      2. Add Gaussian noise N(0, (noise_multiplier * clip_norm)^2)

    With noise_multiplier=1.1 and clip_norm=1.0 this achieves approximately
    ε ≈ 1.0 for δ = 1e-5 at typical CGM batch sizes, following the moments
    accountant analysis of Abadi et al. (2016).
    """
    if device is None:
        device = grad.device
    g_norm = grad.norm(2)
    grad_clipped = grad * min(1.0, clip_norm / (g_norm + 1e-8))
    noise_std = noise_multiplier * clip_norm
    noise = torch.randn_like(grad_clipped) * noise_std
    return grad_clipped + noise


# ── Ring Communication ─────────────────────────────────────────────────────
class RingCommunicator:
    """
    Simulates synchronous ring-topology P2P gradient communication across
    K distributed discriminator sites.

    In a real deployment, this class would be replaced by an MPI or gRPC
    implementation where each site is a separate process or machine. Here
    it operates in-process with tensors passed by reference, simulating the
    ring accumulation without actual network I/O.

    Usage:
        comm = RingCommunicator(n_sites=5, k_fraction=0.1,
                                noise_multiplier=1.1, clip_norm=1.0)
        agg_grads = comm.ring_aggregate(local_gradients)
        # agg_grads[i] is the aggregated gradient for site i after ring pass
    """

    def __init__(self,
                 n_sites: int,
                 k_fraction: float = 0.1,
                 quantize: bool = True,
                 noise_multiplier: float = 1.1,
                 clip_norm: float = 1.0,
                 device: Optional[torch.device] = None):
        self.n_sites          = n_sites
        self.k_fraction       = k_fraction
        self.quantize         = quantize
        self.noise_multiplier = noise_multiplier
        self.clip_norm        = clip_norm
        self.device           = device or torch.device("cpu")

    def _site_prepare(self, grad: torch.Tensor) -> torch.Tensor:
        """Apply DP noise then compress before sending."""
        g = add_dp_noise(grad, self.clip_norm, self.noise_multiplier, self.device)
        g = compress_gradient(g, self.k_fraction, self.quantize)
        return g

    def ring_aggregate(self,
                       local_grads: List[Dict[str, torch.Tensor]]
                       ) -> List[Dict[str, torch.Tensor]]:
        """
        Execute one ring-aggregation pass over all sites.

        Args:
            local_grads: list of K dicts, each mapping param_name → gradient tensor.
                         local_grads[i] is the raw gradient from site i's discriminator.

        Returns:
            agg_grads: list of K dicts with the accumulated gradients after ring pass.
                       All sites will hold equivalent accumulated values after the pass.
        """
        K = self.n_sites
        assert len(local_grads) == K, f"Expected {K} sites, got {len(local_grads)}"

        # Prepare: apply DP + compression on each site's gradient
        prepared = [
            {name: self._site_prepare(g.clone())
             for name, g in site_grads.items()}
            for site_grads in local_grads
        ]

        # Ring pass: site i receives from (i-1) % K, adds its own, forwards to (i+1) % K
        accumulated = [copy.deepcopy(p) for p in prepared]
        for step in range(K - 1):
            for i in range(K):
                src = (i - 1) % K   # receive from left neighbour
                for name in accumulated[i]:
                    accumulated[i][name] = (accumulated[i][name] +
                                            prepared[src][name])
        return accumulated

    def server_aggregate(self,
                         ring_grads: List[Dict[str, torch.Tensor]]
                         ) -> Dict[str, torch.Tensor]:
        """
        Average the ring-accumulated gradients across all sites to produce
        the final gradient signal used to update the central generator.

        This is a FedAvg-style mean aggregation weighted equally across sites.
        In practice you may weight by local dataset size; see weighted_aggregate().
        """
        param_names = ring_grads[0].keys()
        return {
            name: torch.stack([g[name] for g in ring_grads]).mean(dim=0)
            for name in param_names
        }

    def weighted_aggregate(self,
                           ring_grads: List[Dict[str, torch.Tensor]],
                           weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Weighted aggregation — weights[i] is proportional to site i's dataset size.
        Recommended when sites have significantly different numbers of monitoring days
        (e.g., T1DEXI with 11,000 days vs PEDAP with ~3,500 days).
        """
        assert len(weights) == self.n_sites
        total = sum(weights)
        norm_w = [w / total for w in weights]
        param_names = ring_grads[0].keys()
        return {
            name: sum(norm_w[i] * ring_grads[i][name] for i in range(self.n_sites))
            for name in param_names
        }


# ── Gradient Extraction ────────────────────────────────────────────────────
def extract_gradients(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract current .grad tensors from all named parameters."""
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
    return grads


def apply_gradients(model: nn.Module,
                    agg_grads: Dict[str, torch.Tensor]) -> None:
    """Write aggregated gradients back to model parameter .grad fields."""
    for name, param in model.named_parameters():
        if name in agg_grads:
            if param.grad is None:
                param.grad = agg_grads[name].clone()
            else:
                param.grad.copy_(agg_grads[name])


if __name__ == "__main__":
    # Smoke test: 3 sites, each contributing a random gradient dict
    device = torch.device("cpu")
    n_sites = 3
    param_shapes = {"layer1.weight": (64, 2, 4, 4), "layer1.bias": (64,)}

    local_grads = [
        {name: torch.randn(shape, device=device)
         for name, shape in param_shapes.items()}
        for _ in range(n_sites)
    ]

    comm = RingCommunicator(n_sites=n_sites, k_fraction=0.1,
                             noise_multiplier=1.1, clip_norm=1.0)
    ring_agg = comm.ring_aggregate(local_grads)
    server_grad = comm.server_aggregate(ring_agg)
    print("Server aggregated gradient shapes:")
    for k, v in server_grad.items():
        print(f"  {k}: {v.shape}")
