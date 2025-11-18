#!/usr/bin/env python3
"""
Visualize spliced and subsampled features as a spectrogram-like image.
- Loads a Lhotse CutSet (manifest) and reads a target Cut
- Loads features, mean-normalizes, subsamples (every n-th frame), splices context
- Plots 345-dim features (y-axis) vs time (x-axis) as an image
- Draws thin vertical borders around each column (or every k-th column) to emphasize "vector" boundaries
- Optionally overlays per-frame labels at the bottom, with a legend

Requires: lhotse, numpy, matplotlib, torch

Usage:
    python visualize_spliced_features.py --manifest cache/simu_1spk/train_b2_mix100000/cuts_windowed.jsonl.gz --index 0

"""

from __future__ import annotations

import argparse
import os
import math
from typing import Optional

import numpy as np
import torch
from lhotse import CutSet
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from training.ego_dataset import splice as splice_func
from training.ego_dataset import EgoCentricDiarizationDataset
import matplotlib.patches as mpatches


def plot_spliced_features(
    features: np.ndarray,
    label_map: dict[str, int],
    labels: Optional[np.ndarray] = None,
    subsample: int = 10,
    context_size: int = 7,
    border_every: int = 1,
    num_spacing_pixels: int = 0,
    frame_shift: float = 0.01,
    tick_seconds: float = 1.0,
    cmap_name: str = "viridis",
    out_path: Optional[str] = None,
):
    """
    - features: np.ndarray, shape (T, F) raw features (e.g., 23-dim fbank)
    - labels: Optional numpy array with per-frame integer labels (len == T)
    - label_map: Optional mapping from label names to ints
    - subsample: int
    - context_size: int
    - border_every: draw thin border every `border_every` frames (set to 1 to border every frame)
    - num_spacing_pixels: number of blank columns to insert between original frames
    - frame_shift: seconds per frame (before subsampling)
    - out_path: if provided, save output
    """

    device = torch.device("cpu")
    feats_t = torch.from_numpy(features).float().to(device)
    feats_t = feats_t - feats_t.mean(dim=0, keepdim=True)

    if subsample and subsample > 1:
        feats_t = feats_t[::subsample]
        if labels is not None:
            labels = labels[::subsample]

    spliced = splice_func(feats_t, context_size=context_size)

    # spliced: (T_ss, 345)
    spliced_np = spliced.cpu().numpy()

    # Transpose for plotting: y=345, x=time
    im = spliced_np.T
    num_features, num_frames = im.shape

    fig = plt.figure(figsize=(max(8, num_frames / 20), 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[8, 1], hspace=0.05)

    ax = fig.add_subplot(gs[0])
    # If spacing is requested, create a spaced image with blank columns in between
    if num_spacing_pixels > 0:
        spacing = num_spacing_pixels
        frames_spaced = num_frames + spacing * (num_frames - 1)
        im_spaced = np.full((num_features, frames_spaced),
                            np.nan, dtype=im.dtype)
        for j in range(num_frames):
            pos = j * (spacing + 1)
            im_spaced[:, pos] = im[:, j]
        # Mask the spacing columns
        im_for_plot = np.ma.masked_invalid(im_spaced)
        num_frames_plot = frames_spaced
    else:
        im_for_plot = im
        num_frames_plot = num_frames

    # Plot (compute contrast/percentiles on original data without masked columns)
    vmin = float(np.percentile(im, 2))
    vmax = float(np.percentile(im, 98))
    cmap_feat = plt.get_cmap(cmap_name)
    cmap_feat.set_bad(color='white')
    cax = ax.imshow(im_for_plot, aspect="auto", origin="lower",
                    cmap=cmap_feat, vmin=vmin, vmax=vmax)

    ax.set_ylabel("Feature dim (spliced)")
    ax.set_xlabel("Time (s)")

    # Setup time ticks using frame_shift and subsample
    effective_frame_shift = frame_shift * subsample
    if num_frames > 0:
        tick_frames_interval = max(
            1, int(round(tick_seconds / effective_frame_shift)))
        tick_indices = np.arange(0, num_frames, tick_frames_interval)
        if num_spacing_pixels > 0:
            tick_positions = tick_indices * (num_spacing_pixels + 1)
        else:
            tick_positions = tick_indices
        tick_labels = [
            f"{(idx * effective_frame_shift):.2f}s" for idx in tick_indices]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

    # Add borders between each vector (vertical lines between columns)
    if border_every > 0 and num_frames_plot > 0:
        step = border_every * (num_spacing_pixels +
                               1) if num_spacing_pixels > 0 else border_every
        for x in range(0, num_frames_plot + 1, step):
            ax.axvline(x - 0.5, color="k", linewidth=0.3, alpha=0.6)

    # Add a border rectangle around the image
    ax.spines["top"].set_edgecolor("red")
    ax.spines["bottom"].set_edgecolor("red")
    ax.spines["left"].set_edgecolor("red")
    ax.spines["right"].set_edgecolor("red")
    ax.spines["top"].set_linewidth(1.2)

    # Colorbar
    fig.colorbar(cax, ax=ax, fraction=0.02, pad=0.02)

    # Labels (if present)
    if labels is not None:
        ax2 = fig.add_subplot(gs[1], sharex=ax)
        # Represent labels as colored stripes
        label_vals = labels.astype(int)
        unique_labels = np.unique(label_vals)

        inv_label_map = None
        if label_map is not None:
            inv_label_map = {v: k for k, v in label_map.items()} if isinstance(
                list(label_map.keys())[0], str) else label_map

        # Create a ListedColormap with one color per class
        base_cmap = plt.get_cmap('tab10')
        cmap = ListedColormap([base_cmap(i % base_cmap.N)
                              for i in range(len(label_map))])

        # Expand labels to include spacing slots if needed
        if num_spacing_pixels > 0:
            labels_spaced = np.full(num_frames_plot, fill_value=-1, dtype=int)
            for j in range(num_frames):
                pos = j * (num_spacing_pixels + 1)
                labels_spaced[pos] = int(label_vals[j])
            label_im = np.expand_dims(labels_spaced, axis=0)
        else:
            label_im = np.expand_dims(label_vals, axis=0)
        # Build label colormap and add a 'spacing' background color at the end
        n_labels = max(label_map.values()) + 1
        label_colors = [base_cmap(i % base_cmap.N) for i in range(n_labels)]
        label_colors.append((0.9, 0.9, 0.9, 1.0))
        label_colormap = ListedColormap(label_colors)
        display_im = label_im.copy()
        display_im[display_im == -1] = n_labels
        ax2.imshow(display_im, aspect='auto', origin='lower',
                   cmap=label_colormap, vmin=0, vmax=n_labels)
        ax2.get_yaxis().set_visible(False)
        ax2.set_xlabel("Time (s)")

        # Create legend mapping colors to class names
        handles = []

        for key, val in (label_map.items() if isinstance(list(label_map.keys())[0], str) else label_map.items()):
            if isinstance(val, int):
                label_id = val
                label_name = key if isinstance(key, str) else str(key)
            else:
                label_id = val
                label_name = key
            patch = mpatches.Patch(
                color=label_colormap(label_id), label=label_name)
            handles.append(patch)
        ax2.legend(handles=handles, loc='center left',
                   bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        print(f"Saved visualization to {out_path}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize spliced features for a Lhotse CutSet cut')
    parser.add_argument('--manifest', required=True,
                        help='Path to cuts JSONL or JSONL.GZ manifest (CutSet)')
    parser.add_argument('--index', type=int, default=0,
                        help='Index of cut to visualize (default: 0)')
    parser.add_argument('--target_spk', default=None,
                        help='Speaker ID or __none__ for target; default: first speaker in cut or __none__')
    parser.add_argument('--subsample', type=int, default=10,
                        help='Subsample factor (frames)')
    parser.add_argument('--context_size', type=int, default=7,
                        help='Splicing context size (left/right)')
    parser.add_argument('--border_every', type=int, default=1,
                        help='Add thin border every k frames')
    parser.add_argument('--num_spacing_pixels', type=int, default=0,
                        help='Number of blank columns (pixels) to add between each vector')
    parser.add_argument('--frame_shift', type=float, default=0.01,
                        help='Frame shift in seconds before subsampling (e.g., 0.01 for 10ms)')
    parser.add_argument('--tick_seconds', type=float, default=1.0,
                        help='Interval in seconds between x-axis ticks')
    parser.add_argument('--out', default=None,
                        help='Path to save output image (PNG)')
    args = parser.parse_args()

    manifest_path = args.manifest
    assert os.path.exists(
        manifest_path), f"Manifest not found: {manifest_path}"
    print(f"Loading cuts from {manifest_path}...")
    cuts: CutSet = CutSet.from_jsonl_lazy(manifest_path)
    print("Loaded cuts from the manifest.")
    cut = next(iter(cuts))

    features = cut.load_features()
    print(
        f"Loaded features with shape: {features.shape if features is not None else None}")
    if features is None:
        raise RuntimeError("Cut has no features loaded")

    features = np.asarray(features)

    # Find speaker list
    speakers = sorted(set(s.speaker for s in cut.supervisions if s.speaker))

    target_spk = args.target_spk
    if target_spk is None:
        # default: pick first speaker if exists, else '__none__'
        target_spk = speakers[0] if speakers else '__none__'
        if not speakers:
            target_spk = '__none__'

    # Generate labels if possible
    labels = None
    # Use generate_labels_for_cut from the dataset implementation to produce per-frame classes
    if hasattr(EgoCentricDiarizationDataset, 'generate_labels_for_cut') and EgoCentricDiarizationDataset.generate_labels_for_cut is not None:
        label_map = EgoCentricDiarizationDataset.LABEL_MAP
        speakers_list = speakers + [None]
        labels_dict = EgoCentricDiarizationDataset.generate_labels_for_cut(
            cut, speakers_list, label_map)
        key = target_spk if target_spk and target_spk != '__none__' else '__none__'
        labels = labels_dict.get(key)

    out_path = args.out if args.out else f"outputs/visualizations/visualize_{os.path.basename(manifest_path)}_idx{args.index}.png"

    plot_spliced_features(
        features=features,
        labels=labels,
        label_map=EgoCentricDiarizationDataset.LABEL_MAP,
        subsample=args.subsample,
        context_size=args.context_size,
        border_every=args.border_every,
        num_spacing_pixels=args.num_spacing_pixels,
        frame_shift=args.frame_shift,
        tick_seconds=args.tick_seconds,
        out_path=out_path,
    )
