import random
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from tqdm.auto import tqdm

from lhotse import validate, CutSet, SupervisionSet, RecordingSet
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_features
from lhotse.supervision import SupervisionSegment
import numpy as np
from lhotse.utils import Pathlike
Seconds = float


def splice(Y: torch.Tensor, context_size: int = 7) -> torch.Tensor:
    """Feature splicing (concatenate adjacent frames).
    
    Takes context_size frames before + current frame + context_size frames after
    Total frames = 2 * context_size + 1
    Output dimension = input_dim * (2 * context_size + 1)

    Args:
        Y (torch.Tensor): Input features of shape (T, F) 
        context_size (int): Number of frames to concatenate on each side (default: 7)

    Returns:
        torch.Tensor: Spliced features of shape (T, F * (2 * context_size + 1))
    """
    T, F = Y.shape
    
    total_context_frames = 2 * context_size + 1
    spliced_dim = F * total_context_frames
    spliced_Y = torch.zeros(T, spliced_dim, dtype=Y.dtype, device=Y.device)
    
    for t in range(T):
        # Collect context frames
        context_frames = []
        for offset in range(-context_size, context_size + 1):
            idx = t + offset
            if idx < 0:
                # Pad with first frame for frames before the beginning
                context_frames.append(Y[0])
            elif idx >= T:
                # Pad with last frame for frames after the end
                context_frames.append(Y[-1])
            else:
                context_frames.append(Y[idx])
        
        # Concatenate all context frames
        spliced_Y[t] = torch.cat(context_frames, dim=0)
    
    return spliced_Y


def subsample_torch(Y: torch.Tensor, T: torch.Tensor, subsample: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Subsampling utility function for PyTorch tensors.

    Args:
        Y (torch.Tensor): Input features of shape (T, F)
        T (torch.Tensor): Target labels of shape (T,)
        subsample (int): The subsampling factor. Defaults to 10.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of subsampled values.
    """
    Y_ss = Y[::subsample]
    T_ss = T[::subsample]
    return Y_ss, T_ss


def subsample(Y: np.ndarray, T: np.ndarray, subsample: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Simple subsampling utility function

    Args:
        Y (np.ndarray): Input features
        T (np.ndarray): Target speakers (labels)
        subsample (int, optional): the subsampling factor. Defaults to 10.

    Returns:
        Tuple[np.ndarray, np.ndarray]: a tuple of subsampled values. 
    """
    Y_ss = Y[::subsample]
    T_ss = T[::subsample]
    return Y_ss, T_ss


class EgoCentricDiarizationDataset(Dataset):
    """
    A PyTorch Dataset for ego-centric diarization (TS-DIA) that
    pre-computes and caches labels for fast training.
    
    On first run, it will iterate through the entire dataset to
    generate and save labels to a cache file. Subsequent runs
    will load this cache directly.
    """

    # Define the label-to-integer mapping
    LABEL_MAP = {
        'ts': 0,         # target speaker only
        'ts_ovl': 1,     # target speaker overlapping with others
        'others_sgl': 2,  # exactly one non-target speaker
        'others_ovl': 3,  # overlap of non-target speakers
        'ns': 4          # non-speech
    }
    IGNORE_INDEX = CrossEntropyLoss().ignore_index

    def __init__(
        self,
        cuts: CutSet,
        cache_dir: Pathlike,
        chunk_size: float,
        min_enroll_len: float = 1.0,
        max_enroll_len: float = 5.0,
        context_size: int = 7,
        subsampling: int = 10,
        force_recompute: bool = False,
    ) -> None:
        super().__init__()
        validate(cuts)
        self.min_enroll_len = min_enroll_len
        self.max_enroll_len = max_enroll_len
        self.context_size = context_size
        self.subsampling = subsampling
        self.chunk_size = chunk_size
        
        self.cuts = cuts
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "labels.pt"

        # This map is still needed to define the examples
        self.cut_speaker_map: List[Tuple[int, Optional[str]]] = []
        self._build_index_map()
        
        # This will hold the pre-computed labels
        self.cached_labels: List[torch.Tensor] = []

        if self.cache_file.exists() and not force_recompute:
            print(f"Loading pre-computed labels from {self.cache_file}...")
            self._load_labels_from_cache()
        else:
            print("No cache found or recompute forced. Computing labels...")
            self._compute_and_cache_labels()

        print(f"Dataset initialized with {len(self.cut_speaker_map)} examples")

    def _build_index_map(self) -> None:
        """Pre-compute all valid (cut_index, target_speaker) combinations."""
        print("Building dataset index map...")

        for cut_index, cut in enumerate(self.cuts):
            all_speakers_in_cut = sorted(
                set(s.speaker for s in cut.supervisions if s.speaker)
            )
            
            for target_spk_id in all_speakers_in_cut:
                target_supervisions = [
                    s for s in cut.supervisions if s.speaker == target_spk_id
                ]
                
                if target_supervisions:
                    if self._can_get_enrollment(cut, target_spk_id, target_supervisions):
                        self.cut_speaker_map.append((cut_index, target_spk_id))
            
            self.cut_speaker_map.append((cut_index, None))
        
        print(f"Index map built. Total examples: {len(self.cut_speaker_map)}")

    def _load_labels_from_cache(self) -> None:
        """Loads the list of label tensors from the cache file."""
        self.cached_labels = torch.load(self.cache_file)
        if len(self.cached_labels) != len(self.cut_speaker_map):
            raise RuntimeError(
                f"Cache mismatch! Expected {len(self.cut_speaker_map)} labels "
                f"but found {len(self.cached_labels)} in {self.cache_file}. "
                "Please delete the cache file or run with force_recompute=True."
            )

    def _compute_and_cache_labels(self) -> None:
        """
        Generates all labels for the dataset and saves them to a cache file.
        This is a one-time, slow operation.
        """
        print("Generating and caching labels for all examples...")
        all_labels = []
        for cut_index, target_spk_id in tqdm(self.cut_speaker_map, desc="Caching labels"):
            # We must load the cut to generate its labels
            try:
                cut = self.cuts[cut_index]
            except Exception as e:
                print(f"FATAL: Error loading cut at index {cut_index} during cache creation: {e}")
                raise e # Fail fast
                
            labels = self._generate_ego_centric_labels(cut, target_spk_id)
            all_labels.append(labels)
        
        print(f"Saving {len(all_labels)} labels to {self.cache_file}...")
        torch.save(all_labels, self.cache_file)
        self.cached_labels = all_labels
        print("Caching complete.")


    def _can_get_enrollment(
        self,
        cut: Cut,
        target_speaker_id: str,
        target_supervisions: List[SupervisionSegment]
    ) -> bool:
        """Check if we can extract a valid enrollment segment."""
        other_supervisions = [
            s for s in cut.supervisions
            if s.speaker and s.speaker != target_speaker_id
        ]
        continuous_segments = []
        for sup in target_supervisions:
            has_overlap = False
            for other_sup in other_supervisions:
                if (sup.start < other_sup.end and sup.end > other_sup.start):
                    has_overlap = True
                    break
            if not has_overlap and sup.duration > 0:
                continuous_segments.append(sup)
        if not continuous_segments:
            return False
        max_available_len = max(seg.duration for seg in continuous_segments)
        return max_available_len >= self.min_enroll_len

    def __len__(self) -> int:
        """Return the total number of (cut, target_speaker) examples."""
        return len(self.cut_speaker_map)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example by index for standard PyTorch DataLoader.
        
        This method is now very fast as it loads pre-computed labels.
        Per user request, it will BREAK (raise an error) if any data
        is found to be incorrect or unloadable.
        """
        if index >= len(self.cut_speaker_map):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self.cut_speaker_map)}")

        # Get the (cut_index, target_speaker) pair for this index
        cut_index, target_spk_id = self.cut_speaker_map[index]

        # --- ERROR HANDLING: Fail fast as requested ---
        # This will raise an exception if the cut is bad
        cut = self.cuts[cut_index]

        # Extract features using Lhotse's feature loading
        mixture_features = cut.load_features()  # Shape: (T, F)

        if not isinstance(mixture_features, torch.Tensor):
            mixture_features = torch.from_numpy(mixture_features)
        mixture_features = mixture_features.float()

        num_features_dim = mixture_features.shape[1]

        if target_spk_id is not None:
            # Case 1: Real speaker enrollment
            target_supervisions = [
                s for s in cut.supervisions if s.speaker == target_spk_id
            ]
            enroll_cut = self._get_random_enrollment(
                cut, target_spk_id, target_supervisions
            )

            # --- ERROR HANDLING: Fail fast as requested ---
            if enroll_cut is None:
                raise RuntimeError(
                    f"Failed to sample a valid enrollment segment for index {index} "
                    f"(cut: {cut.id}, speaker: {target_spk_id}). "
                    "This can happen if no non-overlapping segment meets min_enroll_len."
                )

            enroll_features = enroll_cut.load_features()  # Shape: (T_enroll, F)
            if not isinstance(enroll_features, torch.Tensor):
                enroll_features = torch.from_numpy(enroll_features)
            enroll_features = enroll_features.float()
        else:
            # Case 2: Zero-enrollment ("+ 1" case)
            enroll_features = torch.zeros(
                1, num_features_dim, dtype=torch.float)

        # --- CACHING: Load pre-computed labels ---
        labels = self.cached_labels[index]
        
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels)
        labels = labels.long()

        # Apply mean normalization per utterance (EEND-style)
        mixture_features = mixture_features - \
            mixture_features.mean(dim=0, keepdim=True)
        enroll_features = enroll_features - \
            enroll_features.mean(dim=0, keepdim=True)

        # Apply subsampling to reduce sequence length
        if self.subsampling > 1:
            mixture_features, labels = subsample_torch(
                mixture_features, labels, subsample=self.subsampling)
            enroll_features, _ = subsample_torch(enroll_features, torch.arange(
                enroll_features.size(0)), subsample=self.subsampling)

        # Apply feature splicing for temporal context (concatenate adjacent frames)
        mixture_features = splice(
            mixture_features, context_size=self.context_size)
        enroll_features = splice(
            enroll_features, context_size=self.context_size)

        return {
            "features": mixture_features,
            "features_lens": torch.tensor(mixture_features.size(0), dtype=torch.long),
            "enroll_features": enroll_features,
            "enroll_features_lens": torch.tensor(enroll_features.size(0), dtype=torch.long),
            "labels": labels,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to handle variable-length sequences with padding.
        
        NOTE: This function no longer checks for `None` items, as __getitem__
        is designed to raise an error instead of returning None.
        """
        # Extract sequences from batch
        features_list = [item["features"] for item in batch]
        enroll_features_list = [item["enroll_features"] for item in batch]
        labels_list = [item["labels"] for item in batch]

        # Get lengths before padding
        features_lens = torch.tensor(
            [feat.size(0) for feat in features_list], dtype=torch.long)
        enroll_features_lens = torch.tensor(
            [feat.size(0) for feat in enroll_features_list], dtype=torch.long)

        # Pad sequences to maximum length in batch
        features_padded = pad_sequence(
            features_list, batch_first=True, padding_value=0.0)
        enroll_features_padded = pad_sequence(
            enroll_features_list, batch_first=True, padding_value=0.0)
        labels_padded = pad_sequence(
            labels_list, batch_first=True, padding_value=EgoCentricDiarizationDataset.IGNORE_INDEX)

        return {
            "features": features_padded,  # (B, T_max, F)
            "features_lens": features_lens,  # (B,)
            "enroll_features": enroll_features_padded,  # (B, T_enroll_max, F)
            "enroll_features_lens": enroll_features_lens,  # (B,)
            "labels": labels_padded,  # (B, T_max)
        }

    def _get_random_enrollment(
        self,
        cut: Cut,
        target_speaker_id: str,
        target_supervisions: List[SupervisionSegment]
    ) -> Optional[Cut]:
        """
        Randomly sample a continuous enrollment segment from the target speaker's speech.
        """
        other_supervisions = [
            s for s in cut.supervisions
            if s.speaker and s.speaker != target_speaker_id
        ]
        continuous_segments = []
        for sup in target_supervisions:
            has_overlap = False
            for other_sup in other_supervisions:
                if (sup.start < other_sup.end and sup.end > other_sup.start):
                    has_overlap = True
                    break
            if not has_overlap and sup.duration > 0:
                continuous_segments.append(sup)
        if not continuous_segments:
            return None

        enroll_len = random.uniform(self.min_enroll_len, self.max_enroll_len)
        valid_segments = [
            s for s in continuous_segments if s.duration >= enroll_len]
        if not valid_segments:
            longest_segment = max(continuous_segments,
                                  key=lambda s: s.duration)
            if longest_segment.duration < self.min_enroll_len:
                return None
            enroll_len = min(longest_segment.duration, self.max_enroll_len)
            valid_segments = [longest_segment]

        selected_segment = random.choice(valid_segments)
        if selected_segment.start < 0 or selected_segment.end > cut.duration:
            return None

        max_start_offset = selected_segment.duration - enroll_len
        start_offset = random.uniform(
            0, max_start_offset) if max_start_offset > 0 else 0
        enroll_start = selected_segment.start + start_offset
        
        if enroll_start < 0:
            enroll_start = 0.0
        max_possible_start = cut.duration - enroll_len
        if enroll_start > max_possible_start:
            enroll_start = max(0.0, max_possible_start)
        if enroll_start < 0 or enroll_start >= cut.duration:
            return None
        if enroll_len <= 0 or enroll_start + enroll_len > cut.duration:
            return None

        enroll_cut = cut.truncate(
            offset=enroll_start,
            duration=enroll_len,
            preserve_id=False
        )
        return enroll_cut

    def _generate_ego_centric_labels(
        self,
        cut: Cut,
        target_speaker_id: Optional[str]
    ) -> torch.Tensor:
        """
        Generates the 1D categorical label sequence Y for a given cut
        from the perspective of target_speaker_id.
        """
        num_frames = cut.num_frames
        if num_frames is None or num_frames == 0:
            return torch.empty(0, dtype=torch.long)

        frame_shift = cut.frame_shift
        if frame_shift is None:
            frame_shift = 0.01

        target_mask = torch.zeros(num_frames, dtype=torch.bool)
        other_speaker_count = torch.zeros(num_frames, dtype=torch.int32)

        all_speakers: set[str] = {
            s.speaker for s in cut.supervisions if s.speaker}

        if target_speaker_id is not None:
            other_speaker_ids: set[str] = all_speakers - {target_speaker_id}
            for sup in cut.supervisions:
                if sup.speaker == target_speaker_id:
                    start_frame, end_frame = self._supervision_to_frames(
                        sup, num_frames, frame_shift)
                    target_mask[start_frame:end_frame] = True
        else:
            other_speaker_ids: set[str] = all_speakers
            
        for other_spk_id in other_speaker_ids:
            other_spk_mask = torch.zeros(num_frames, dtype=torch.bool)
            for sup in cut.supervisions:
                if sup.speaker == other_spk_id:
                    start_frame, end_frame = self._supervision_to_frames(
                        sup, num_frames, frame_shift)
                    other_spk_mask[start_frame:end_frame] = True
            other_speaker_count += other_spk_mask.int()

        labels = torch.full(
            (num_frames,),
            fill_value=self.LABEL_MAP['ns'],
            dtype=torch.long
        )

        labels[(target_mask == True) & (
            other_speaker_count == 0)] = self.LABEL_MAP['ts']
        labels[(target_mask == True) & (other_speaker_count > 0)
               ] = self.LABEL_MAP['ts_ovl']
        labels[(target_mask == False) & (other_speaker_count == 1)
               ] = self.LABEL_MAP['others_sgl']
        labels[(target_mask == False) & (other_speaker_count > 1)
               ] = self.LABEL_MAP['others_ovl']

        return labels

    def _supervision_to_frames(self, sup: SupervisionSegment, num_frames: int, frame_shift: float) -> Tuple[int, int]:
        """Helper to convert supervision start/end times to frame indices."""
        start_frame = round(sup.start / frame_shift)
        end_frame = start_frame + round(sup.duration / frame_shift)
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        return start_frame, end_frame