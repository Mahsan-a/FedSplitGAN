"""
fed_split_gan.py
----------------
Federated Split GAN training orchestrator.

One training round:
  1. Server samples z ~ N(0,I) → Generator G produces X_syn
  2. Server broadcasts X_syn to all K client sites
  3. Each site:
       a. Samples a mini-batch X_real from local data
       b. Runs forward pass through local Discriminator D_k on (X_real, X_syn)
       c. Computes discriminator loss + WGAN-GP gradient penalty (local)
       d. Backpropagates through D_k
       e. Extracts ∇D_k  (gradient of D_k w.r.t. X_syn)
       f. Applies DP noise + top-k compression
  4. P2P ring communication aggregates compressed ∇D across all sites
  5. Server applies aggregated ∇G to update Generator via Adam
  6. Sites update their own Discriminators using local Adam steps

The split is between X_syn (server-side) and the discriminators (client-side).
The generator never sees raw patient data. Each discriminator only sees
its own site's real data plus the shared synthetic samples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .generator      import Generator, CGANGenerator, sample_noise
from .discriminator  import (Discriminator, ConditionalDiscriminator,
                               discriminator_loss, generator_loss,
                               gradient_penalty)
from .communication  import (RingCommunicator, extract_gradients,
                               apply_gradients)


# ── Config ─────────────────────────────────────────────────────────────────
@dataclass
class FedSplitGANConfig:
    # Architecture
    latent_dim     : int   = 128
    n_classes      : int   = 4          # risk groups; 0 = unconditional
    conditional    : bool  = False

    # Training
    n_rounds       : int   = 500        # federated communication rounds
    n_disc_steps   : int   = 3          # discriminator updates per round
    batch_size     : int   = 16
    lr_gen         : float = 1e-4
    lr_disc        : float = 2e-4
    betas          : tuple = (0.5, 0.999)
    loss_mode      : str   = "hinge"    # "bce" | "hinge" | "wgan"
    lambda_gp      : float = 10.0       # WGAN-GP penalty weight

    # Federated / P2P
    k_fraction     : float = 0.1        # top-k gradient sparsification
    noise_multiplier: float = 1.1       # Gaussian DP noise multiplier
    clip_norm      : float = 1.0        # DP gradient clipping norm
    weighted_agg   : bool  = True       # weight aggregation by site size

    # Logging
    log_every      : int   = 20
    save_every     : int   = 100
    save_dir       : str   = "checkpoints"


# ── Federated Site ─────────────────────────────────────────────────────────
class FederatedSite:
    """
    Represents one distributed clinical site in the federated training.

    Holds: local discriminator, local optimizer, local DataLoader.
    Exposes: one_step() — performs n_disc_steps discriminator updates given
             X_syn from the server, and returns ∇D w.r.t. X_syn for
             transmission to the ring communicator.
    """

    def __init__(self,
                 site_id     : int,
                 dataloader  : DataLoader,
                 config      : FedSplitGANConfig,
                 device      : torch.device,
                 site_weight : float = 1.0):
        self.site_id      = site_id
        self.loader       = dataloader
        self.cfg          = config
        self.device       = device
        self.site_weight  = site_weight    # proportional to local dataset size
        self._iter        = iter(dataloader)

        if config.conditional:
            self.disc = ConditionalDiscriminator(
                n_classes=config.n_classes).to(device)
        else:
            self.disc = Discriminator().to(device)

        self.optimizer = optim.Adam(
            self.disc.parameters(),
            lr=config.lr_disc,
            betas=config.betas
        )

    def _next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cycle through local DataLoader."""
        try:
            x, y = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            x, y = next(self._iter)
        return x.to(self.device), y.to(self.device)

    def one_step(self,
                 x_syn: torch.Tensor,
                 labels: Optional[torch.Tensor] = None
                 ) -> Dict[str, torch.Tensor]:
        """
        Perform n_disc_steps discriminator updates on (x_real, x_syn).

        Returns the gradient of the discriminator's output w.r.t. x_syn —
        this is the signal sent back to the central server to update G.
        The discriminator's own weights are updated locally and never shared.

        Args:
            x_syn:  (batch, 2, 96, 288) synthetic scalograms from generator
            labels: (batch,)  risk-group labels — only used in conditional mode

        Returns:
            grad_signal: dict param_name → grad tensor for ring communication
        """
        self.disc.train()
        x_syn_local = x_syn.detach()   # detach: D update doesn't flow back to G

        for _ in range(self.cfg.n_disc_steps):
            x_real, y_real = self._next_batch()
            self.optimizer.zero_grad()

            # Forward
            if self.cfg.conditional and labels is not None:
                y = labels[:x_real.size(0)].to(self.device)
                real_scores = self.disc(x_real, y)
                fake_scores = self.disc(x_syn_local, y)
            else:
                real_scores = self.disc(x_real)
                fake_scores = self.disc(x_syn_local)

            d_loss = discriminator_loss(real_scores, fake_scores,
                                         mode=self.cfg.loss_mode)

            # Gradient penalty — applied locally, before gradient transmission
            if self.cfg.loss_mode == "wgan":
                gp = gradient_penalty(self.disc, x_real, x_syn_local,
                                       self.device, self.cfg.lambda_gp)
                d_loss = d_loss + gp

            d_loss.backward()
            self.optimizer.step()

        # ── Compute ∇D w.r.t. X_syn for generator update ──────────────────
        # Run one more forward pass with grad tracking on x_syn to obtain
        # the gradient that will update the generator.
        x_syn_probe = x_syn.requires_grad_(True)
        if self.cfg.conditional and labels is not None:
            fake_s = self.disc(x_syn_probe, labels[:x_syn.size(0)].to(self.device))
        else:
            fake_s = self.disc(x_syn_probe)

        g_loss_local = generator_loss(fake_s, mode=self.cfg.loss_mode)
        g_loss_local.backward()

        # x_syn.grad is the signal we transmit — it tells the generator
        # how to improve its outputs from this site's perspective.
        grad_signal = {"x_syn_grad": x_syn_probe.grad.detach().clone()}
        return grad_signal


# ── Central Server / Orchestrator ──────────────────────────────────────────
class FedSplitGANTrainer:
    """
    Central server orchestrating the federated GAN training loop.

    The server holds the generator and its optimizer. It broadcasts synthetic
    samples to all sites each round, collects gradient signals through the
    ring communicator, and updates the generator.
    """

    def __init__(self,
                 sites    : List[FederatedSite],
                 config   : FedSplitGANConfig,
                 device   : torch.device):
        self.sites  = sites
        self.cfg    = config
        self.device = device
        self.K      = len(sites)

        # Central generator
        if config.conditional:
            self.generator = CGANGenerator(
                latent_dim=config.latent_dim,
                n_classes=config.n_classes).to(device)
        else:
            self.generator = Generator(latent_dim=config.latent_dim).to(device)

        self.gen_opt = optim.Adam(
            self.generator.parameters(),
            lr=config.lr_gen,
            betas=config.betas
        )

        self.communicator = RingCommunicator(
            n_sites=self.K,
            k_fraction=config.k_fraction,
            noise_multiplier=config.noise_multiplier,
            clip_norm=config.clip_norm,
            device=device
        )

        os.makedirs(config.save_dir, exist_ok=True)
        self.history: Dict[str, List[float]] = {"g_loss": []}

    def _sample_batch_size(self) -> int:
        return self.cfg.batch_size

    def train(self) -> None:
        """Main federated training loop."""
        for rnd in range(1, self.cfg.n_rounds + 1):
            bs = self._sample_batch_size()

            # ── 1. Server generates synthetic samples ─────────────────────
            self.generator.train()
            self.gen_opt.zero_grad()

            z = sample_noise(bs, self.cfg.latent_dim, self.device)

            if self.cfg.conditional:
                # Sample balanced class labels for conditional generation
                labels = torch.randint(0, self.cfg.n_classes, (bs,), device=self.device)
                x_syn  = self.generator(z, labels)
            else:
                labels = None
                x_syn  = self.generator(z)

            # ── 2–3. Broadcast to sites; sites perform discriminator steps ─
            site_grads: List[Dict[str, torch.Tensor]] = []
            for site in self.sites:
                grad_signal = site.one_step(x_syn, labels)
                site_grads.append(grad_signal)

            # ── 4. Ring communication: P2P gradient aggregation ───────────
            ring_agg = self.communicator.ring_aggregate(site_grads)

            if self.cfg.weighted_agg:
                weights = [s.site_weight for s in self.sites]
                agg_grad = self.communicator.weighted_aggregate(ring_agg, weights)
            else:
                agg_grad = self.communicator.server_aggregate(ring_agg)

            # ── 5. Update generator using aggregated gradient signal ───────
            # The aggregated gradient acts as the error signal from all sites
            # combined — x_syn.grad equivalent for the generator update.
            # We use the aggregated x_syn gradient to compute generator loss.
            with torch.no_grad():
                x_syn_agg_grad = agg_grad["x_syn_grad"]

            # Recompute generator output for a clean backward pass
            self.gen_opt.zero_grad()
            z2 = sample_noise(bs, self.cfg.latent_dim, self.device)
            if self.cfg.conditional:
                x_syn2 = self.generator(z2, labels)
            else:
                x_syn2 = self.generator(z2)

            # Apply surrogate loss: dot product with aggregated gradient signal
            # This is equivalent to one step of the REINFORCE / straight-through
            # estimator in federated GAN literature.
            g_loss = (x_syn2 * x_syn_agg_grad).mean()
            g_loss.backward()
            self.gen_opt.step()

            self.history["g_loss"].append(g_loss.item())

            # ── Logging ───────────────────────────────────────────────────
            if rnd % self.cfg.log_every == 0:
                print(f"[Round {rnd:4d}/{self.cfg.n_rounds}]  "
                      f"g_loss={g_loss.item():.4f}")

            # ── Checkpointing ─────────────────────────────────────────────
            if rnd % self.cfg.save_every == 0:
                self._save_checkpoint(rnd)

        print("[FedSplitGAN] Training complete.")
        self._save_checkpoint(self.cfg.n_rounds, final=True)

    def _save_checkpoint(self, rnd: int, final: bool = False) -> None:
        tag = "final" if final else f"round{rnd}"
        path = os.path.join(self.cfg.save_dir, f"gen_{tag}.pt")
        torch.save({
            "round"            : rnd,
            "generator_state"  : self.generator.state_dict(),
            "optimizer_state"  : self.gen_opt.state_dict(),
            "history"          : self.history,
            "config"           : self.cfg,
        }, path)
        print(f"[checkpoint] saved → {path}")

    @torch.no_grad()
    def generate_synthetic(self,
                            n_samples: int,
                            label: Optional[int] = None) -> torch.Tensor:
        """
        Generate n_samples synthetic CGM scalograms post-training.

        Args:
            n_samples: number of samples to generate
            label:     risk group label (0=rare, 1=occ, 2=freq, 3=high)
                       Set to 0 to generate rare-event class for augmentation.
                       Ignored if unconditional model.
        Returns:
            x_syn: (n_samples, 2, 96, 288) tensor, values in [-1, 1]
        """
        self.generator.eval()
        z = sample_noise(n_samples, self.cfg.latent_dim, self.device)
        if self.cfg.conditional and label is not None:
            lbl = torch.full((n_samples,), label,
                              dtype=torch.long, device=self.device)
            return self.generator(z, lbl)
        return self.generator(z)
