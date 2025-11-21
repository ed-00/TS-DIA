#!/usr/bin/env python3
"""Load and inspect a .pt diagnostic file created by Trainer (diag_nan_batches).

Usage:
  python scripts/diagnose_diag_pt.py path/to/bad_batch_epoch0_step1743_batch1743.pt

What it prints:
 - Top-level keys and types
 - Stats for tensors: shape, dtype, min/max/mean/std, count of NaN/Inf
 - Small sample prints (first few values)
 - Helpful checks for out-of-range labels or very large values

This is intentionally dependency-light: requires torch and numpy (both available in this repo).
"""
import argparse
import textwrap
from pathlib import Path
import math
import numpy as np
import torch


def is_tensor(x):
    return isinstance(x, torch.Tensor)


def to_numpy(x):
    if is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return None


def print_tensor_stats(name, t):
    arr = to_numpy(t)
    if arr is None:
        print(f"  {name}: Not a tensor/array (type={type(t)}) -> {repr(t)[:200]}")
        return

    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")

    # Flatten for numeric stats
    try:
        flat = arr.ravel()
        # cast to float64 for stable stats when necessary
        if flat.size == 0:
            print("    (empty tensor)")
            return
        finite_mask = np.isfinite(flat)
        n_total = flat.size
        n_finite = int(finite_mask.sum())
        n_inf = int(np.isinf(flat).sum())
        n_nan = int(np.isnan(flat).sum())
        print(f"    total={n_total}, finite={n_finite}, inf={n_inf}, nan={n_nan}")

        if n_finite > 0:
            finite_vals = flat[finite_mask]
            print(
                f"    min={float(finite_vals.min()):.6g}, max={float(finite_vals.max()):.6g}, mean={float(finite_vals.mean()):.6g}, std={float(finite_vals.std()):.6g}"
            )
            # show some percentiles for scale
            pcts = np.percentile(finite_vals, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
            pct_line = ", ".join([f"p{p}:{v:.6g}" for p, v in zip([0.1, 1, 5, 25, 50, 75, 95, 99, 99.9], pcts)])
            print(f"    percentiles: {pct_line}")

        # show a small sample
        sample_count = min(10, n_total)
        if sample_count > 0:
            print("    sample:", np.array2string(flat[:sample_count], precision=6, threshold=10))
    except Exception as e:
        print(f"    Failed computing stats: {e}")


def inspect_diag(path: Path):
    print(f"Loading: {path}\n")
    data = torch.load(str(path), map_location="cpu")

    if not isinstance(data, dict):
        print(f"File contents are not a dict, got {type(data)}. Raw repr:\n{repr(data)[:1000]}")
        return

    print("Top-level keys and types:")
    for k, v in data.items():
        t = type(v)
        if is_tensor(v):
            print(f"- {k}: tensor {v.shape} dtype={v.dtype}")
        else:
            print(f"- {k}: {t}")

    # Print common fields if present
    int_fields = ["epoch", "step", "batch_idx"]
    for f in int_fields:
        if f in data:
            print(f"{f}: {data[f]}")

    # Logits stats included sometimes
    if "logits_min" in data or "logits_max" in data:
        print("\nSaved logits summary:")
        print(f"  logits_min: {data.get('logits_min')}")
        print(f"  logits_max: {data.get('logits_max')}")

    # Diagnose features
    if "features" in data:
        print("\nFEATURES:")
        print_tensor_stats("features", data["features"])

        # Check for extremely large magnitude
        arr = to_numpy(data["features"])
        if arr is not None and arr.size > 0:
            max_abs = float(np.abs(arr).max())
            if max_abs > 1e6:
                print(f"    ⚠️ Extremely large feature magnitude detected: max_abs={max_abs:.6g}")
            if np.any(np.isfinite(arr) == False):
                print("    ⚠️ Non-finite values detected inside features (NaN/Inf)")

    # Diagnose labels
    if "labels" in data:
        print("\nLABELS:")
        print_tensor_stats("labels", data["labels"])
        lab = to_numpy(data["labels"])
        if lab is not None and lab.size > 0:
            # Basic label checks: integer type? range
            if np.issubdtype(lab.dtype, np.floating):
                print("    Warning: labels are float-type; expected integer class indices for classification tasks.")
            try:
                amin = int(np.nanmin(lab))
                amax = int(np.nanmax(lab))
                print(f"    label range: {amin} .. {amax}")
            except Exception:
                pass

            # If labels look like class indices, show uniques
            try:
                uniques = np.unique(lab)
                if uniques.size <= 20:
                    print(f"    unique labels ({len(uniques)}): {uniques}")
                else:
                    print(f"    unique label count: {len(uniques)} (too many to display)")
            except Exception:
                pass

    # Print any other tensors nearby
    for key, val in data.items():
        if key in ("features", "labels", "logits_min", "logits_max", "epoch", "step", "batch_idx"):
            continue
        if is_tensor(val):
            print(f"\nOTHER TENSOR: {key}")
            print_tensor_stats(key, val)

    print("\nDone.\n")


def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
        Load a .pt file saved by Trainer diagnostics and print useful stats.
        Example: python scripts/diagnose_diag_pt.py outputs/checkpoints/softmax_pretraining_large_pos/diag_nan_batches/bad_batch_epoch0_step1743_batch1743.pt
        """)
    )
    parser.add_argument("path", type=Path, help="Path to .pt file")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: {args.path} does not exist")
        raise SystemExit(2)

    inspect_diag(args.path)


if __name__ == "__main__":
    main()
