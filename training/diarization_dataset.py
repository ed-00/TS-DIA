#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Custom dataset classes for speaker diarization.

This module contains specialized dataset classes for speaker diarization tasks,
including on-the-fly feature extraction datasets.
"""

from typing import Optional

import torch
from lhotse import CutSet
from lhotse.dataset.collation import collate_matrices
from torch.nn import CrossEntropyLoss


class DiarizationOnTheFlyDataset(torch.utils.data.Dataset):
    """
    On-the-fly feature extraction dataset for speaker diarization.

    This dataset performs feature extraction on-demand and builds speaker activity
    matrices aligned to the extracted features for diarization tasks.

    Args:
        cuts: Lhotse CutSet containing audio cuts with supervisions
        input_strategy: Lhotse input strategy for feature extraction
        frame_shift: Frame shift in seconds for alignment (default: 0.01)
        min_speaker_dim: Minimum speaker dimension to pad to (optional)
    """

    def __init__(
        self,
        cuts: CutSet,
        input_strategy,
        frame_shift: float = 0.01,
        min_speaker_dim: Optional[int] = None
    ):
        self.cuts = cuts
        self.input_strategy = input_strategy
        self.frame_shift = frame_shift
        self.min_speaker_dim = min_speaker_dim

    def __getitem__(self, cuts_batch: CutSet):
        feats, feat_lens, *rest = self.input_strategy(cuts_batch)

        # Build per-cut speaker activity matrices in frame domain using frame_shift
        masks = []
        for idx, cut in enumerate(cuts_batch):
            num_frames = int(feat_lens[idx].item())
            speakers = sorted(set(s.speaker for s in cut.supervisions))

            # Handle speaker indexing
            speaker_to_idx = {spk: i for i, spk in enumerate(speakers)}
            num_speakers = len(speakers)

            # Apply min_speaker_dim regardless of whether speakers exist
            if self.min_speaker_dim is not None:
                num_speakers = max(num_speakers, int(self.min_speaker_dim))

            # Create mask with proper dimensions
            mask = torch.zeros((num_speakers, num_frames), dtype=torch.long)

            # Fill in speaker activity (only if speakers exist)
            if speakers:
                for sup in cut.supervisions:
                    spk_idx = speaker_to_idx[sup.speaker]
                    st = int(round(max(0.0, sup.start) / self.frame_shift))
                    et = int(round(min(cut.duration, sup.end) / self.frame_shift))
                    st = max(0, min(st, num_frames))
                    et = max(0, min(et, num_frames))
                    if et > st:
                        mask[spk_idx, st:et] = 1

            # Transpose to [num_frames, num_speakers] for collate_matrices
            # (collate_matrices pads along the first dimension)
            masks.append(mask.T)

        # Collate masks: input is list of [num_frames, num_speakers], output is [batch, max_frames, num_speakers]
        speaker_activity = collate_matrices(
            masks, padding_value=CrossEntropyLoss().ignore_index
        )

        # Transpose to [batch, num_speakers, max_frames] to match expected format
        speaker_activity = speaker_activity.transpose(1, 2)

        return {
            "features": feats,
            "features_lens": feat_lens,
            "speaker_activity": speaker_activity,
        }