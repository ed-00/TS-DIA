import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Optional, List, Tuple
from tqdm.auto import tqdm

from lhotse import validate, CutSet
from lhotse.cut import MonoCut
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
        accelerator: Accelerator,
        context_size: int = 7,
        subsampling: int = 10,

    ) -> None:
        super().__init__()
        validate(cuts)

        self.accelerator = accelerator
        self.context_size = context_size
        self.subsampling = subsampling
        self.cuts = cuts.to_eager()
        self.cut_speaker_map: List[Tuple[str, Optional[str]]] = []
        
        # Try to load index map from cache first
        self._load_or_build_index_map()

    def _load_or_build_index_map(self) -> None:
        """
        Load index map from cache if available, otherwise build it.
        """
        # Check if all cuts have cached index maps
        all_cached = True
        cached_map: List[Tuple[str, Optional[str]]] = []
        
        for cut in list(self.cuts):
            if cut.custom and 'ego_index_map' in cut.custom:
                cached_entries = cut.custom['ego_index_map']
                cached_map.extend(cached_entries)
            else:
                all_cached = False
                break
        
        if all_cached and cached_map:
            self.accelerator.print(
                f"✓ Loaded index map from cache: {len(cached_map)} speaker examples"
            )
            self.cut_speaker_map = cached_map
        else:
            self.accelerator.print(
                f"→ Cache miss: Building index map from scratch..."
            )
            self._build_index_map()
            self.accelerator.print(
                f"✓ Built index map: {len(self.cut_speaker_map)} speaker examples"
            )

    def _build_index_map(self) -> None:
        """
        Builds the cut_speaker_map with a single, fast loop.
        """
        for cut in iter(self.cuts):
            all_speakers_in_cut = sorted(
                set(s.speaker for s in cut.supervisions if s.speaker)
            )

            for target_spk_id in all_speakers_in_cut:
                self.cut_speaker_map.append((cut.id, target_spk_id))

            self.cut_speaker_map.append((cut.id, None))

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
        # Ensure we have a MonoCut (all cuts should be MonoCut type)
        assert isinstance(cut, MonoCut), f"Expected MonoCut, got {type(cut)}"

        mixture_features = cut.load_features()

        mixture_features = torch.from_numpy(mixture_features)
        mixture_features = mixture_features.float()

        mixture_features = mixture_features - \
            mixture_features.mean(dim=0, keepdim=True)

        # Load labels from cache (should always exist after preprocessing)
        speaker_key = target_spk_id if target_spk_id else "__none__"
        
        if cut.custom and 'ego_labels' in cut.custom:
            cached_labels = cut.custom['ego_labels'].get(speaker_key)
            if cached_labels is not None:
                labels = torch.from_numpy(cached_labels)
            else:
                raise RuntimeError(
                    f"Labels not found for cut {cut_id}, speaker {speaker_key}. "
                    "Labels should be pre-computed during dataset loading."
                )
        else:
            raise RuntimeError(
                f"No ego_labels found in cut.custom for cut {cut_id}. "
                "Labels should be pre-computed during dataset loading."
            )

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
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function.
        Now only collates features and labels.
        """
        features_list = [item["features"] for item in batch]
        labels_list = [item["labels"]for item in batch]

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
        }

    @staticmethod
    def _generate_ego_centric_labels_static(
        cut: MonoCut,
        target_speaker_id: Optional[str],
        label_map: Dict[str, int],
    ) -> torch.Tensor:
        """
        Generates the 1D categorical label sequence Y using
        lhotse.speakers_feature_mask.
        """
        num_frames = cut.num_frames
        if num_frames is None or num_frames == 0:
            return torch.empty(0, dtype=torch.long)

        all_speakers_in_cut = sorted(
            set(s.speaker for s in cut.supervisions if s.speaker)
        )
        if not all_speakers_in_cut:
            return torch.full((num_frames,), fill_value=label_map['ns'], dtype=torch.long)

        speaker_to_idx_map = {spk: i for i,
                              spk in enumerate(all_speakers_in_cut)}
        mask = cut.speakers_feature_mask(speaker_to_idx_map=speaker_to_idx_map)

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

        return torch.from_numpy(labels)
