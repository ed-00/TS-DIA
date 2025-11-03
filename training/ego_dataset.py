import random

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Optional, List, Tuple

from lhotse import validate, CutSet, SupervisionSet, RecordingSet
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_features
from lhotse.supervision import SupervisionSegment
import numpy as np
Seconds = float


def _count_frames(data_len: int, size: int, step: int)-> int:
    """Simple utility function for counting frames.

    Args:
        data_len (int): number of samples 
        size (int): window size in samples 
        step (int): step size

    Returns:
        int: floored number of frames
    """
    # no padding at edges, last remaining samples are ignored
    # github.com/hitachi-speech/EEND
    return int((data_len - size + step) / step)


def _gen_frame_indices(
    data_length: int,
    size: int = 2000,
    step: int = 2000,
    use_last_samples: bool = False,
    label_delay: int = 0,
    subsampling: int = 1
):
    """Need modification TODO

    Args:
        data_length (_type_): _description_
        size (int, optional): _description_. Defaults to 2000.
        step (int, optional): _description_. Defaults to 2000.
        use_last_samples (bool, optional): _description_. Defaults to False.
        label_delay (int, optional): _description_. Defaults to 0.
        subsampling (int, optional): _description_. Defaults to 1.

    Yields:
        _type_: _description_
    """
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step - subsampling * label_delay > 0:
            yield (i + 1) * step, data_length


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
    extracts target speaker enrollment chunks *from the cut itself*.

    Feature Processing Pipeline (following EEND):
    1. Input: F-dimensional acoustic features (e.g., MFCC, filterbank) 
    2. Chunking: Fixed-size segments of chunk_size seconds (provides temporal context)
    3. Mean normalization per utterance
    4. Subsampling: Factor of subsampling → reduced sequence length
    
    For each cut, it generates N new training examples (one per speaker).
    Each example contains:
    1. The features for the *full mixture* (X) - spliced and subsampled
    2. The features for *only the target speaker's speech* (the enrollment data) - spliced and subsampled
    3. The 1D categorical labels (Y) from that speaker's perspective - subsampled

    Output batch dictionary:
    .. code-block::
        {
            'features': (B x T_sub x F) tensor (the full mixture)
            'features_lens': (B,) tensor
            'enroll_features': (B x T_enroll_sub x F) tensor (target speaker's speech)
            'enroll_features_lens': (B,) tensor
            'labels': (B x T_sub) tensor (the 5-class ego-centric labels)
        }
        
    Where:
        B = batch size
        T_sub = subsampled sequence length (original_length // subsampling)
        F = input feature dimension (e.g., 80 for log-mel filterbank)
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
        chunk_size: float,
        min_enroll_len: float = 1.0,
        max_enroll_len: float = 5.0,
        context_size: int = 7,
        subsampling: int = 10,
    ) -> None:
        super().__init__()
        validate(cuts)
        self.min_enroll_len = min_enroll_len
        self.max_enroll_len = max_enroll_len
        self.context_size = context_size
        self.subsampling = subsampling
        self.chunk_size = chunk_size
        
        # Cuts should already be chunked by the DatasetManager
        self.cuts = cuts

        # Pre-compute all (cut, target_speaker) pairs for proper __len__
        self.examples: List[Tuple[Cut, str]] = []
        self._build_examples()
        
        print(f"Dataset initialized with {len(self.examples)} examples")
        self.cuts.describe()

    def _build_examples(self) -> None:
        """Pre-compute all valid (cut, target_speaker) combinations."""
        cuts_list = list(self.cuts)  # Convert CutSet to list for iteration
        for cut in cuts_list:
            # Get unique speakers from supervisions
            all_speakers_in_cut = sorted(
                set(s.speaker for s in cut.supervisions if s.speaker)
            )
            
            for target_spk_id in all_speakers_in_cut:
                # Filter supervisions for target speaker only
                target_supervisions = [
                    s for s in cut.supervisions if s.speaker == target_spk_id
                ]
                
                if target_supervisions:
                    # Check if we can get a valid enrollment segment
                    if self._can_get_enrollment(cut, target_spk_id, target_supervisions):
                        self.examples.append((cut, target_spk_id))

    def _can_get_enrollment(
        self,
        cut: Cut,
        target_speaker_id: str,
        target_supervisions: List[SupervisionSegment]
    ) -> bool:
        """Check if we can extract a valid enrollment segment."""
        # Get all other speakers' supervisions to detect overlap
        other_supervisions = [
            s for s in cut.supervisions
            if s.speaker and s.speaker != target_speaker_id
        ]

        # Find continuous segments (no overlap with other speakers)
        continuous_segments = []
        for sup in target_supervisions:
            # Check if this supervision overlaps with any other speaker
            has_overlap = False
            for other_sup in other_supervisions:
                # Check for temporal overlap
                if (sup.start < other_sup.end and sup.end > other_sup.start):
                    has_overlap = True
                    break

            if not has_overlap and sup.duration > 0:
                continuous_segments.append(sup)

        if not continuous_segments:
            return False

        # Check if any segment is long enough for minimum enrollment
        max_available_len = max(seg.duration for seg in continuous_segments)
        return max_available_len >= self.min_enroll_len

    def __len__(self) -> int:
        """Return the total number of (cut, target_speaker) examples."""
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example by index for standard PyTorch DataLoader.
        
        Args:
            index: Integer index into the examples list
            
        Returns:
            Dictionary containing features, enrollment features, and labels for a single example
        """
        if index >= len(self.examples):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.examples)}")
        
        # Get the (cut, target_speaker) pair for this index
        cut, target_spk_id = self.examples[index]
        
        # Get target speaker supervisions
        target_supervisions = [
            s for s in cut.supervisions if s.speaker == target_spk_id
        ]

        # Get enrollment segment
        enroll_cut = self._get_random_enrollment(
            cut, target_spk_id, target_supervisions
        )

        if enroll_cut is None:
            # Skip this example - return next valid example instead
            return self.__getitem__((index + 1) % len(self.examples))

        # Extract features using Lhotse's feature loading
        mixture_features = cut.load_features()  # Shape: (T, F)
        enroll_features = enroll_cut.load_features()  # Shape: (T_enroll, F)

        # Generate ego-centric labels
        labels = self._generate_ego_centric_labels(cut, target_spk_id)  # Shape: (T,)

        # Convert to torch tensors and ensure correct types
        if not isinstance(mixture_features, torch.Tensor):
            mixture_features = torch.from_numpy(mixture_features)
        mixture_features = mixture_features.float()
        
        if not isinstance(enroll_features, torch.Tensor):
            enroll_features = torch.from_numpy(enroll_features)
        enroll_features = enroll_features.float()
        
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels)
        labels = labels.long()

        # Apply mean normalization per utterance (EEND-style)
        mixture_features = mixture_features - mixture_features.mean(dim=0, keepdim=True)
        enroll_features = enroll_features - enroll_features.mean(dim=0, keepdim=True)

        # Apply subsampling to reduce sequence length
        if self.subsampling > 1:
            mixture_features, labels = subsample_torch(mixture_features, labels, subsample=self.subsampling)
            enroll_features, _ = subsample_torch(enroll_features, torch.arange(enroll_features.size(0)), subsample=self.subsampling)

        # Apply feature splicing for temporal context (concatenate adjacent frames)
        mixture_features = splice(mixture_features, context_size=self.context_size)
        enroll_features = splice(enroll_features, context_size=self.context_size)

        return {
            "features": mixture_features,  # (T_sub, F * (2*context_size + 1)) - spliced features
            "features_lens": torch.tensor(mixture_features.size(0), dtype=torch.long),
            "enroll_features": enroll_features,  # (T_enroll_sub, F * (2*context_size + 1))
            "enroll_features_lens": torch.tensor(enroll_features.size(0), dtype=torch.long),
            "labels": labels,  # (T_sub,)
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to handle variable-length sequences with padding.
        
        Args:
            batch: List of individual examples from __getitem__
            
        Returns:
            Collated and padded batch dictionary
        """
        # Extract sequences from batch
        features_list = [item["features"] for item in batch]
        enroll_features_list = [item["enroll_features"] for item in batch]
        labels_list = [item["labels"] for item in batch]
        
        # Get lengths before padding
        features_lens = torch.tensor([feat.size(0) for feat in features_list], dtype=torch.long)
        enroll_features_lens = torch.tensor([feat.size(0) for feat in enroll_features_list], dtype=torch.long)
        
        # Pad sequences to maximum length in batch
        features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
        enroll_features_padded = pad_sequence(enroll_features_list, batch_first=True, padding_value=0.0)
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=EgoCentricDiarizationDataset.IGNORE_INDEX)
        
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

        The enrollment segment:
        - Has a random length between min_enroll_len and max_enroll_len
        - Starts at a random position within a continuous speech segment
        - Contains only the target speaker (no overlap with other speakers)
        - Contains no non-speech regions

        Returns None if no suitable segment can be found.
        """
        # Get all other speakers' supervisions to detect overlap
        other_supervisions = [
            s for s in cut.supervisions
            if s.speaker and s.speaker != target_speaker_id
        ]

        # Find continuous segments (no overlap with other speakers)
        continuous_segments = []
        for sup in target_supervisions:
            # Check if this supervision overlaps with any other speaker
            has_overlap = False
            for other_sup in other_supervisions:
                # Check for temporal overlap
                if (sup.start < other_sup.end and sup.end > other_sup.start):
                    has_overlap = True
                    break

            if not has_overlap and sup.duration > 0:
                continuous_segments.append(sup)

        if not continuous_segments:
            return None

        # Try to find a segment long enough for enrollment
        # Sample a random enrollment length
        enroll_len = random.uniform(self.min_enroll_len, self.max_enroll_len)

        # Filter segments that are long enough
        valid_segments = [
            s for s in continuous_segments if s.duration >= enroll_len]

        if not valid_segments:
            # If no segment is long enough, use the longest available segment
            # and adjust enroll_len to fit
            longest_segment = max(continuous_segments,
                                  key=lambda s: s.duration)
            if longest_segment.duration < self.min_enroll_len:
                # Even the longest segment is too short
                return None
            enroll_len = min(longest_segment.duration, self.max_enroll_len)
            valid_segments = [longest_segment]

        # Randomly select a segment
        selected_segment = random.choice(valid_segments)

        # Validate segment boundaries
        if selected_segment.start < 0 or selected_segment.end > cut.duration:
            # Segment extends outside cut boundaries, skip this cut
            return None

        # Randomly select a start position within the segment
        max_start_offset = selected_segment.duration - enroll_len
        start_offset = random.uniform(
            0, max_start_offset) if max_start_offset > 0 else 0

        # Calculate offset from cut start
        # Note: selected_segment.start is already relative to the cut's coordinate system
        enroll_start = selected_segment.start + start_offset
        
        # Ensure offset is non-negative and within cut boundaries
        if enroll_start < 0:
            # Debug info for negative offset issue
            print(f"Warning: Negative enrollment start {enroll_start:.3f}. "
                  f"Segment start: {selected_segment.start:.3f}, "
                  f"Start offset: {start_offset:.3f}, "
                  f"Cut duration: {cut.duration:.3f}")
            enroll_start = 0.0
        
        # Ensure we don't exceed cut duration
        max_possible_start = cut.duration - enroll_len
        if enroll_start > max_possible_start:
            enroll_start = max(0.0, max_possible_start)

        # Final safety check before truncate
        if enroll_start < 0 or enroll_start >= cut.duration:
            print(f"Error: Invalid enroll_start {enroll_start} for cut duration {cut.duration}")
            return None
            
        if enroll_len <= 0 or enroll_start + enroll_len > cut.duration:
            print(f"Error: Invalid enroll_len {enroll_len} for cut duration {cut.duration} and start {enroll_start}")
            return None

        # Trim the cut to the enrollment region
        enroll_cut = cut.truncate(
            offset=enroll_start,
            duration=enroll_len,
            preserve_id=False
        )

        return enroll_cut

    def _generate_ego_centric_labels(
        self,
        cut: Cut,
        target_speaker_id: str
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
            # Default frame shift if not available (common values: 0.01 for 10ms, 0.02 for 20ms)
            frame_shift = 0.01

        target_mask = torch.zeros(num_frames, dtype=torch.bool)
        other_speaker_count = torch.zeros(num_frames, dtype=torch.int32)

        # Get other speaker IDs from supervisions
        all_speakers: set[str] = {
            s.speaker for s in cut.supervisions if s.speaker}
        other_speaker_ids: set[str] = all_speakers - {target_speaker_id}

        # Create target speaker mask
        for sup in cut.supervisions:
            if sup.speaker == target_speaker_id:
                start_frame, end_frame = self._supervision_to_frames(
                    sup, num_frames, frame_shift)
                target_mask[start_frame:end_frame] = True

        # Create other speaker masks and sum them
        for other_spk_id in other_speaker_ids:
            other_spk_mask = torch.zeros(num_frames, dtype=torch.bool)
            # Filter supervisions for this specific speaker
            for sup in cut.supervisions:
                if sup.speaker == other_spk_id:
                    start_frame, end_frame = self._supervision_to_frames(
                        sup, num_frames, frame_shift)
                    other_spk_mask[start_frame:end_frame] = True
            other_speaker_count += other_spk_mask.int()

        # Map masks to the 5 categorical labels
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
