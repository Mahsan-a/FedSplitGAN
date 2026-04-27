"""
data_utils.py
-------------
CGM data loading, dual-wavelet CWT scalogram computation, and PyTorch
Dataset/DataLoader construction for the FedSplitGAN framework.

Input:  Raw CGM CSV files — one row per reading, columns: [patient_id, timestamp, glucose_mg_dl]
Output: Scalogram tensors of shape (96, 288, 2) — [freq_scales, time_points, wavelet_channels]
        Channel 0: Mexican Hat (Ricker) CWT  — temporal localization, rapid transitions
        Channel 1: Morlet CWT               — frequency resolution, slow metabolic trends
"""

import os
import numpy as np
import pandas as pd
import pywt
from scipy.signal import savgol_filter
from typing import List, Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader


# ── Constants ──────────────────────────────────────────────────────────────
GLUCOSE_READINGS_PER_DAY = 288          # 5-min CGM, 24 hrs
N_SCALES                 = 96           # frequency scales for CWT
HYPO_THRESHOLD           = 70.0         # mg/dL
MAX_GAP_HOURS            = 4.0          # days with more missing data are excluded
SAVGOL_WINDOW            = 7
SAVGOL_POLYORDER         = 3
GAUSSIAN_SIGMA           = 2.0
GAUSSIAN_HALF_WIN        = 11           # half of window=23
FLUCTUATION_THRESHOLD_FACTOR = 0.5     # theta = sigma_BG / 2


# ── CWT Scales ─────────────────────────────────────────────────────────────
def build_scales(n_scales: int = N_SCALES,
                 min_period_min: float = 15.0,
                 max_period_min: float = 1440.0,
                 sampling_min: float = 5.0) -> np.ndarray:
    """
    Linearly spaced scales corresponding to periods from min_period_min to
    max_period_min (in minutes) at 5-minute CGM sampling resolution.
    """
    min_scale = min_period_min / sampling_min
    max_scale = max_period_min / sampling_min
    return np.linspace(min_scale, max_scale, n_scales)


SCALES = build_scales()


# ── TSGF Preprocessing ─────────────────────────────────────────────────────
def tsgf_preprocess(glucose: np.ndarray) -> Optional[np.ndarray]:
    """
    Temporal Smoothing and Gap Filling (TSGF) — four-step pipeline that
    prepares a raw CGM day-trace for wavelet transformation by filling gaps
    while preserving clinically relevant glycemic excursions.

    Returns None if >4 hours of data are missing (day excluded).

    Steps:
        1. Linear interpolation for short gaps
        2. Gaussian kernel smoothing for robustness on longer gaps
        3. Clinical feature preservation: reintroduce fluctuations > sigma_BG/2
        4. Savitzky-Golay filtering to remove CWT-disruptive sharp edges
    """
    n = len(glucose)
    if n != GLUCOSE_READINGS_PER_DAY:
        glucose = _resample_to_288(glucose)

    # Check missingness
    missing_mask = np.isnan(glucose)
    max_gap = _max_consecutive_nans(missing_mask)
    if max_gap > (MAX_GAP_HOURS * 60 / 5):   # 4 hrs = 48 readings
        return None

    # Step 1: linear interpolation
    x = np.arange(n)
    valid = ~missing_mask
    if valid.sum() < n:
        glucose = np.interp(x, x[valid], glucose[valid])

    # Step 2: Gaussian kernel smoothing
    smooth = _gaussian_smooth(glucose, sigma=GAUSSIAN_SIGMA, half_win=GAUSSIAN_HALF_WIN)

    # Step 3: preserve significant glycemic excursions
    sigma_bg = np.std(glucose[valid]) if valid.sum() > 1 else np.std(glucose)
    theta = sigma_bg * FLUCTUATION_THRESHOLD_FACTOR
    fluctuations = glucose - smooth
    preserved = smooth.copy()
    sig_idx = np.abs(fluctuations) > theta
    preserved[sig_idx] += fluctuations[sig_idx]

    # Step 4: Savitzky-Golay smoothing to remove residual sharp edges
    result = savgol_filter(preserved, window_length=SAVGOL_WINDOW,
                           polyorder=SAVGOL_POLYORDER)
    return result.astype(np.float32)


def _gaussian_smooth(x: np.ndarray, sigma: float, half_win: int) -> np.ndarray:
    """Weighted moving average with Gaussian kernel."""
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    offsets = np.arange(-half_win, half_win + 1)
    weights_template = np.exp(-(offsets ** 2) / (2 * sigma ** 2))
    for k in range(n):
        idx = k + offsets
        valid = (idx >= 0) & (idx < n)
        w = weights_template[valid]
        w /= w.sum()
        out[k] = np.dot(w, x[idx[valid]])
    return out


def _max_consecutive_nans(mask: np.ndarray) -> int:
    """Return the length of the longest run of True values in mask."""
    max_run, run = 0, 0
    for v in mask:
        run = run + 1 if v else 0
        max_run = max(max_run, run)
    return max_run


def _resample_to_288(glucose: np.ndarray) -> np.ndarray:
    """Linear resample to exactly 288 points."""
    from scipy.interpolate import interp1d
    x_old = np.linspace(0, 1, len(glucose))
    x_new = np.linspace(0, 1, GLUCOSE_READINGS_PER_DAY)
    f = interp1d(x_old, glucose, bounds_error=False, fill_value="extrapolate")
    return f(x_new).astype(np.float32)


# ── Dual CWT ───────────────────────────────────────────────────────────────
def compute_dual_cwt(glucose: np.ndarray,
                     scales: np.ndarray = SCALES,
                     normalize: bool = True) -> np.ndarray:
    """
    Apply Mexican Hat (Ricker) and Morlet CWTs to a preprocessed glucose
    trace and return a (96, 288, 2) scalogram tensor.

    Channel 0: Mexican Hat — detects rapid transitions (meal spikes, hypo onset)
    Channel 1: Morlet      — detects slow sustained trends (overnight drift, circadian)
    """
    # Mexican Hat (Ricker) CWT
    mh_coefs, _ = pywt.cwt(glucose, scales, wavelet='mexh')    # (96, 288)

    # Morlet CWT — pywt uses 'cmor' with bandwidth and centre frequency
    # cmor1.5-1.0 satisfies the admissibility condition (centre freq ≈ 6 rad)
    mo_coefs, _ = pywt.cwt(glucose, scales, wavelet='cmor1.5-1.0')
    mo_coefs = np.abs(mo_coefs)    # take magnitude for real-valued representation

    mh_coefs = mh_coefs.astype(np.float32)
    mo_coefs = mo_coefs.astype(np.float32)

    if normalize:
        mh_coefs = _minmax_norm(mh_coefs)
        mo_coefs = _minmax_norm(mo_coefs)

    scalogram = np.stack([mh_coefs, mo_coefs], axis=-1)   # (96, 288, 2)
    return scalogram


def _minmax_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + eps)


# ── CGM Day Loader ─────────────────────────────────────────────────────────
def load_cgm_csv(path: str,
                 glucose_col: str = "glucose_mg_dl",
                 patient_col: str = "patient_id",
                 time_col: str = "timestamp") -> Dict[str, List[np.ndarray]]:
    """
    Load a CGM CSV file and return a dict mapping patient_id → list of
    daily glucose arrays (one per monitoring day).

    Expected CSV schema:
        patient_id | timestamp (YYYY-MM-DD HH:MM) | glucose_mg_dl
    """
    df = pd.read_csv(path, parse_dates=[time_col])
    df = df.sort_values([patient_col, time_col])
    df["date"] = df[time_col].dt.date

    patient_days: Dict[str, List[np.ndarray]] = {}
    for pid, pgroup in df.groupby(patient_col):
        days = []
        for date, dgroup in pgroup.groupby("date"):
            g = dgroup[glucose_col].values.astype(np.float32)
            days.append(g)
        patient_days[str(pid)] = days
    return patient_days


def build_scalogram_dataset(csv_path: str,
                             hypoglycemia_labels: Optional[str] = None
                             ) -> Tuple[List[np.ndarray], List[int]]:
    """
    End-to-end pipeline: CSV → TSGF → dual CWT → scalogram list.

    Args:
        csv_path:            Path to CGM CSV file.
        hypoglycemia_labels: Optional path to a CSV with columns
                             [patient_id, date, label] where label ∈ {0,1}.
                             Used for supervised downstream training only;
                             the GAN training itself is unsupervised.

    Returns:
        scalograms: List of (96,288,2) float32 arrays.
        labels:     List of int labels (0/1). Empty list if no label file.
    """
    patient_days = load_cgm_csv(csv_path)
    label_map = {}
    if hypoglycemia_labels is not None:
        ldf = pd.read_csv(hypoglycemia_labels)
        for _, row in ldf.iterrows():
            label_map[(str(row["patient_id"]), str(row["date"]))] = int(row["label"])

    scalograms, labels = [], []
    skipped = 0
    for pid, days in patient_days.items():
        for i, glucose in enumerate(days):
            clean = tsgf_preprocess(glucose)
            if clean is None:
                skipped += 1
                continue
            scalo = compute_dual_cwt(clean)
            scalograms.append(scalo)
            labels.append(label_map.get((pid, str(i)), -1))

    print(f"[data_utils] Built {len(scalograms)} scalograms | skipped {skipped} days (>4hr gap)")
    return scalograms, labels


# ── PyTorch Dataset ────────────────────────────────────────────────────────
class CGMScalogramDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-computed CGM scalograms.

    Args:
        scalograms: List of (96, 288, 2) numpy arrays.
        labels:     List of int labels (-1 = unlabeled, used in GAN training).
        transform:  Optional callable applied to each scalogram tensor.
    """

    def __init__(self,
                 scalograms: List[np.ndarray],
                 labels: Optional[List[int]] = None,
                 transform=None):
        self.scalograms = scalograms
        self.labels     = labels if labels is not None else [-1] * len(scalograms)
        self.transform  = transform

    def __len__(self) -> int:
        return len(self.scalograms)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # (96, 288, 2) → (2, 96, 288) for PyTorch conv layers
        x = torch.from_numpy(self.scalograms[idx]).permute(2, 0, 1).float()
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[idx]


def make_dataloader(dataset: CGMScalogramDataset,
                    batch_size: int = 16,
                    shuffle: bool = True,
                    num_workers: int = 2) -> DataLoader:
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=True)


# ── CLI entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, pickle

    parser = argparse.ArgumentParser(description="Build CGM scalogram dataset")
    parser.add_argument("--data_dir",  required=True, help="Directory of CGM CSV files")
    parser.add_argument("--out_dir",   required=True, help="Output directory for scalograms")
    parser.add_argument("--labels",    default=None,  help="Optional hypoglycemia label CSV")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for fname in sorted(os.listdir(args.data_dir)):
        if not fname.endswith(".csv"):
            continue
        site_name = fname.replace(".csv", "")
        csv_path  = os.path.join(args.data_dir, fname)
        print(f"\n[{site_name}] processing...")
        scalos, labs = build_scalogram_dataset(csv_path, args.labels)
        out = {"scalograms": scalos, "labels": labs, "site": site_name}
        with open(os.path.join(args.out_dir, f"{site_name}_scalograms.pkl"), "wb") as f:
            pickle.dump(out, f)
        print(f"[{site_name}] saved {len(scalos)} scalograms → {args.out_dir}")
