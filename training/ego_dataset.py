import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Optional, List, Tuple, cast
from pathlib import Path
from tqdm.auto import tqdm

from lhotse import validate, CutSet
from lhotse.cut import MonoCut, MixedCut, Cut
import numpy as np
from accelerate import Accelerator


Seconds = float


def splice(Y: torch.Tensor, context_size: int = 7) -> torch.Tensor:
    """Feature splicing (concatenate adjacent frames)."""
    T, F = Y.shape
    total_context_frames = 2 * context_size + 1
    spliced_dim = F * total_context_frames
    spliced_Y = torch.zeros(T, spliced_dim, dtype=Y.dtype, device=Y.device)
    for t in range(T):
        context_frames = []
        for offset in range(-context_size, context_size + 1):
            idx = t + offset
            if idx < 0:
                context_frames.append(Y[0])
            elif idx >= T:
                context_frames.append(Y[-1])
            else:
                context_frames.append(Y[idx])
        spliced_Y[t] = torch.cat(context_frames, dim=0)
    return spliced_Y


def subsample_torch(Y: torch.Tensor, T: torch.Tensor, subsample: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Subsampling utility function for PyTorch tensors."""
    Y_ss = Y[::subsample]
    T_ss = T[::subsample]
    return Y_ss, T_ss


class EgoCentricDiarizationDataset(Dataset):
    """
    A fast "on-the-fly" dataset with NO enrollment.

    - INSTANT STARTUP: Builds the index with a simple,
      fast, single-threaded loop.
    - NO ENROLLMENT: Only returns features and labels.
    - HEAVY GETITEM: All processing is done in __getitem__.

    *** WARNING: Must be used with a DataLoader with
    *** num_workers > 0 to avoid extreme bottlenecks.
    """

    LABEL_MAP = {
        'ts': 0, 'ts_ovl': 1, 'others_sgl': 2,
        'others_ovl': 3, 'ns': 4
    }
    IGNORE_INDEX = CrossEntropyLoss().ignore_index

    def __init__(
        self,
        cuts: CutSet,
        context_size: int = 7,
        subsampling: int = 10,
        validate_cuts: bool = True,

    ) -> None:
        super().__init__()
        self.context_size = context_size
        self.subsampling = subsampling
  

        self.cuts = cuts.to_eager()
        self.cut_speaker_map: List[Tuple[str,
                                         Optional[str]]] = self._build_index_map()

    def _build_index_map(self) -> List[Tuple[str, Optional[str]]]:
        """
        Builds the cut_speaker_map with a single, fast loop.
        """
        cut_speaker_map = []
        for idx, cut in enumerate(iter(self.cuts)):
            all_speakers_in_cut = sorted(
                set(s.speaker for s in cut.supervisions if s.speaker)
            )

            for target_spk_id in all_speakers_in_cut:
                cut_speaker_map.append((cut.id, target_spk_id))

            cut_speaker_map.append((cut.id, None))

        return cut_speaker_map

    def __len__(self) -> int:
        return len(self.cut_speaker_map)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.
        Loads features and labels from cache (labels should be pre-computed).
        """
        if index >= len(self.cut_speaker_map):
            raise IndexError("Index out of range")

        cut_id, target_spk_id = self.cut_speaker_map[index]

        cut = self.cuts[cut_id]

        cut = cast(MixedCut, cut)

        mixture_features = cut.load_features()

        mixture_features = torch.from_numpy(mixture_features)
        mixture_features = mixture_features.float()

        mixture_features = mixture_features - \
            mixture_features.mean(dim=0, keepdim=True)

        # Generate labels on-the-fly
        speaker_key = target_spk_id if target_spk_id else "__none__"
        all_speakers_in_cut = sorted(
            set(s.speaker for s in cut.supervisions if s.speaker))
        speaker_ids = all_speakers_in_cut + [None]
        labels_dict = self.generate_labels_for_cut(
            cut, speaker_ids, self.LABEL_MAP)
        np_labels = labels_dict[speaker_key]
        labels = torch.from_numpy(np_labels).long()

        if self.subsampling > 1:
            mixture_features, labels = subsample_torch(
                mixture_features, labels, subsample=self.subsampling
            )

        mixture_features = splice(
            mixture_features, context_size=self.context_size
        )

        return {
            "features": mixture_features,
            "features_lens": torch.tensor(mixture_features.size(0), dtype=torch.long),
            "labels": labels,
            "speaker_id": target_spk_id if target_spk_id else "__none__",
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function.
        Now only collates features and labels.
        """
        features_list = [item["features"] for item in batch]
        labels_list = [item["labels"]for item in batch]
        speaker_ids = [item["speaker_id"] for item in batch]

        features_lens = torch.tensor(
            [feat.size(0) for feat in features_list], dtype=torch.long)

        features_padded = pad_sequence(
            features_list, batch_first=True, padding_value=0.0)
        labels_padded = pad_sequence(
            labels_list, batch_first=True, padding_value=EgoCentricDiarizationDataset.IGNORE_INDEX)

        return {
            "features": features_padded,
            "features_lens": features_lens,
            "labels": labels_padded,
            "speaker_ids": speaker_ids,
        }

    @staticmethod
    def generate_labels_for_cut(
        cut: Cut,
        speaker_ids: List[Optional[str]],
        label_map: Dict[str, int],
    ) -> Dict[str, np.ndarray]:
        """
        Generate ego-centric labels for all target speakers in a cut.
        Returns dict mapping speaker_id to labels array.
        """
        labels_dict = {}
        num_frames = cut.num_frames

        if num_frames is None or num_frames == 0:
            for spk_id in speaker_ids:
                speaker_key = spk_id if spk_id else "__none__"
                labels_dict[speaker_key] = np.empty(0, dtype=np.int64)
            return labels_dict

        all_speakers_in_cut = sorted(
            set(s.speaker for s in cut.supervisions if s.speaker)
        )

        if not all_speakers_in_cut:
            for spk_id in speaker_ids:
                speaker_key = spk_id if spk_id else "__none__"
                labels_dict[speaker_key] = np.full(
                    num_frames, fill_value=label_map['ns'], dtype=np.int64
                )
            return labels_dict

        speaker_to_idx_map = {spk: i for i,
                              spk in enumerate(all_speakers_in_cut)}
        mask = cut.speakers_feature_mask(speaker_to_idx_map=speaker_to_idx_map)

        # Generate labels for each target speaker
        for target_speaker_id in speaker_ids:
            speaker_key = target_speaker_id if target_speaker_id else "__none__"

            if target_speaker_id is None:
                target_mask = np.zeros(num_frames, dtype=np.int32)
                other_speaker_count = np.sum(mask, axis=0)
            else:
                target_idx = speaker_to_idx_map[target_speaker_id]
                target_mask = mask[target_idx]
                other_indices = [
                    i for spk, i in speaker_to_idx_map.items()
                    if spk != target_speaker_id
                ]
                if other_indices:
                    other_speaker_count = np.sum(mask[other_indices], axis=0)
                else:
                    other_speaker_count = np.zeros(num_frames, dtype=np.int32)

            labels = np.full(
                num_frames, fill_value=label_map['ns'], dtype=np.int64)
            labels[(target_mask == 1) & (
                other_speaker_count == 0)] = label_map['ts']
            labels[(target_mask == 1) & (
                other_speaker_count > 0)] = label_map['ts_ovl']
            labels[(target_mask == 0) & (other_speaker_count == 1)
                   ] = label_map['others_sgl']
            labels[(target_mask == 0) & (other_speaker_count > 1)
                   ] = label_map['others_ovl']

            labels_dict[speaker_key] = labels

        return labels_dict
