"""
train.py
--------
Entry point for FedSplitGAN training on distributed CGM scalogram datasets.

Each CSV file in --data_dir is treated as one federated client (clinical site).
The script constructs one FederatedSite per CSV, builds a FedSplitGANTrainer,
and runs the full federated training loop.

Example usage:
    python train.py \
        --data_dir   data/cgm_csvs \
        --save_dir   checkpoints \
        --n_rounds   500 \
        --conditional \
        --loss_mode  hinge

Mapping of CSV files to clinical sites (matches your thesis datasets):
    t1dexi.csv      → Site 0  (491 adults,     weight ∝ 11,000 days)
    t1dexip.csv     → Site 1  (227 adolescents, weight ∝  1,682 days)
    cl3.csv         → Site 2  (168 patients,    weight ∝ 25,991 days)
    cl5.csv         → Site 3  (100 patients,    weight ∝ 15,466 days)
    city.csv        → Site 4  (149 patients,    weight ∝ 19,108 days)
    pedap.csv       → Site 5  (98 patients,     weight ∝ 19,524 days)
    aidet1d.csv     → Site 6  (82 patients,     weight ∝ 24,566 days)
"""

import argparse
import pickle
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.data_utils    import (CGMScalogramDataset, make_dataloader,
                                build_scalogram_dataset)
from src.fed_split_gan import FedSplitGANConfig, FedSplitGANTrainer, FederatedSite


def parse_args():
    p = argparse.ArgumentParser(description="FedSplitGAN — CGM scalogram generation")
    p.add_argument("--data_dir",      required=True,  help="Directory of CGM CSV files")
    p.add_argument("--save_dir",      default="checkpoints")
    p.add_argument("--scalogram_dir", default=None,
                   help="Pre-computed scalogram .pkl files (skips CWT recomputation)")
    p.add_argument("--n_rounds",      type=int,   default=500)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--latent_dim",    type=int,   default=128)
    p.add_argument("--lr_gen",        type=float, default=1e-4)
    p.add_argument("--lr_disc",       type=float, default=2e-4)
    p.add_argument("--loss_mode",     default="hinge", choices=["bce","hinge","wgan"])
    p.add_argument("--lambda_gp",     type=float, default=10.0)
    p.add_argument("--k_fraction",    type=float, default=0.1,
                   help="Top-k gradient sparsification fraction")
    p.add_argument("--noise_mult",    type=float, default=1.1,
                   help="Gaussian DP noise multiplier")
    p.add_argument("--clip_norm",     type=float, default=1.0)
    p.add_argument("--conditional",   action="store_true",
                   help="Use conditional GAN (requires hypoglycemia frequency labels)")
    p.add_argument("--log_every",     type=int, default=20)
    p.add_argument("--save_every",    type=int, default=100)
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


def load_site_data(data_dir: str,
                   scalogram_dir: str | None,
                   batch_size: int) -> list:
    """
    Load each CSV / pre-computed scalogram file as one FederatedSite dataset.
    Returns list of (DataLoader, site_weight) tuples.
    """
    sites_data = []

    if scalogram_dir is not None:
        # Fast path: load pre-computed scalograms
        pkl_files = sorted(f for f in os.listdir(scalogram_dir) if f.endswith(".pkl"))
        for pkl in pkl_files:
            with open(os.path.join(scalogram_dir, pkl), "rb") as f:
                obj = pickle.load(f)
            ds = CGMScalogramDataset(obj["scalograms"], obj["labels"])
            dl = make_dataloader(ds, batch_size=batch_size, shuffle=True)
            sites_data.append((dl, len(ds)))
            print(f"  Loaded {pkl:35s}  →  {len(ds):5d} scalograms")
    else:
        # Slow path: compute CWT from CSVs on the fly
        csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
        for csv in csv_files:
            scalos, labels = build_scalogram_dataset(os.path.join(data_dir, csv))
            ds = CGMScalogramDataset(scalos, labels)
            dl = make_dataloader(ds, batch_size=batch_size, shuffle=True)
            sites_data.append((dl, len(ds)))
            print(f"  Processed {csv:33s}  →  {len(ds):5d} scalograms")

    return sites_data


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train.py] Device: {device}")

    # ── Config ──────────────────────────────────────────────────────────────
    cfg = FedSplitGANConfig(
        latent_dim      = args.latent_dim,
        conditional     = args.conditional,
        n_rounds        = args.n_rounds,
        batch_size      = args.batch_size,
        lr_gen          = args.lr_gen,
        lr_disc         = args.lr_disc,
        loss_mode       = args.loss_mode,
        lambda_gp       = args.lambda_gp,
        k_fraction      = args.k_fraction,
        noise_multiplier= args.noise_mult,
        clip_norm       = args.clip_norm,
        log_every       = args.log_every,
        save_every      = args.save_every,
        save_dir        = args.save_dir,
    )

    # ── Load site data ───────────────────────────────────────────────────────
    print("\n[train.py] Loading client site data...")
    sites_data = load_site_data(args.data_dir, args.scalogram_dir, args.batch_size)

    if len(sites_data) == 0:
        raise ValueError(f"No data found in {args.data_dir}")

    print(f"\n[train.py] {len(sites_data)} federated sites registered.\n")

    # ── Build FederatedSite objects ──────────────────────────────────────────
    sites = [
        FederatedSite(
            site_id     = i,
            dataloader  = dl,
            config      = cfg,
            device      = device,
            site_weight = float(n_samples)
        )
        for i, (dl, n_samples) in enumerate(sites_data)
    ]

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = FedSplitGANTrainer(sites=sites, config=cfg, device=device)

    print(f"[train.py] Starting federated training: {cfg.n_rounds} rounds, "
          f"{len(sites)} sites\n")
    trainer.train()

    # ── Sample synthetic scalograms ──────────────────────────────────────────
    print("\n[train.py] Generating 64 synthetic scalograms (rare-event class)...")
    label = 0 if cfg.conditional else None   # 0 = rare-event class
    x_syn = trainer.generate_synthetic(n_samples=64, label=label)
    print(f"[train.py] Synthetic output shape: {x_syn.shape}")  # (64, 2, 96, 288)

    import torch
    out_path = os.path.join(args.save_dir, "synthetic_samples.pt")
    torch.save(x_syn.cpu(), out_path)
    print(f"[train.py] Saved synthetic samples → {out_path}")


if __name__ == "__main__":
    main()
