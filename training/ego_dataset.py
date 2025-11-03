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
    2. Mean normalization per utterance
    3. Splicing: ±context_size frames → F * (2*context_size + 1) dimensional features
    4. Subsampling: Factor of subsampling → reduced sequence length
    
    For each cut, it generates N new training examples (one per speaker).
    Each example contains:
    1. The features for the *full mixture* (X) - spliced and subsampled
    2. The features for *only the target speaker's speech* (the enrollment data) - spliced and subsampled
    3. The 1D categorical labels (Y) from that speaker's perspective - subsampled

    Output batch dictionary:
    .. code-block::
        {
            'features': (B x T_sub x F_spliced) tensor (the full mixture, spliced)
            'features_lens': (B,) tensor
            'enroll_features': (B x T_enroll_sub x F_spliced) tensor (target speaker's speech, spliced)
            'enroll_features_lens': (B,) tensor
            'labels': (B x T_sub) tensor (the 5-class ego-centric labels)
        }
        
    Where:
        B = batch size
        T_sub = subsampled sequence length (original_length // subsampling)
        F_spliced = input_feature_dim * (2 * context_size + 1)
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

    def __getitem__(self, batch_or_indices) -> Dict[str, torch.Tensor]:
        """
        Takes either a CutSet (from Lhotse samplers) or list of indices and returns a collated batch.
        
        Args:
            batch_or_indices: Either a CutSet from Lhotse samplers or List[int] of indices
        """
        mixture_cuts: List[Cut] = []
        enroll_cuts: List[Cut] = []
        labels_list: List[torch.Tensor] = []

        # Handle different input types
        if isinstance(batch_or_indices, CutSet):
            # Standard Lhotse pattern: receive CutSet, extract (cut, speaker) pairs
            cuts_list = list(batch_or_indices)
            examples_to_process = []
            
            for cut in cuts_list:
                # Get all speakers for this cut
                all_speakers_in_cut = sorted(
                    set(s.speaker for s in cut.supervisions if s.speaker)
                )
                
                # For each speaker, create an example
                for target_spk_id in all_speakers_in_cut:
                    target_supervisions = [
                        s for s in cut.supervisions if s.speaker == target_spk_id
                    ]
                    
                    if target_supervisions and self._can_get_enrollment(cut, target_spk_id, target_supervisions):
                        examples_to_process.append((cut, target_spk_id))
        else:
            # List of indices pattern: use pre-computed examples
            examples_to_process = [self.examples[idx] for idx in batch_or_indices]

        # Process examples
        for cut, target_spk_id in examples_to_process:
            # Get target speaker supervisions
            target_supervisions = [
                s for s in cut.supervisions if s.speaker == target_spk_id
            ]

            # Get enrollment segment
            enroll_cut = self._get_random_enrollment(
                cut, target_spk_id, target_supervisions
            )

            if enroll_cut is None:
                continue

            # Add cuts and labels
            mixture_cuts.append(cut)
            enroll_cuts.append(enroll_cut)
            labels_list.append(
                self._generate_ego_centric_labels(cut, target_spk_id)
            )

        if not mixture_cuts:
            # Batch was empty
            return {
                "features": torch.empty(0, 0, 0),
                "features_lens": torch.empty(0, dtype=torch.long),
                "enroll_features": torch.empty(0, 0, 0),
                "enroll_features_lens": torch.empty(0, dtype=torch.long),
                "labels": torch.empty(0, 0, dtype=torch.long),
            }

        # Collate mixture features: (B x T x F)
        features, features_lens = collate_features(
            CutSet.from_cuts(mixture_cuts)
        )

        # Collate enrollment features: (B x T_enroll x F)
        enroll_features, enroll_features_lens = collate_features(
            CutSet.from_cuts(enroll_cuts)
        )

        # Process features with splicing and subsampling
        processed_features = []
        processed_enroll_features = []
        processed_labels = []

        for i in range(len(mixture_cuts)):
            # Get valid feature sequences
            mix_feat = features[i, :features_lens[i], :]  # (T, F)
            enr_feat = enroll_features[i, :enroll_features_lens[i], :]  # (T_enroll, F)
            labels = labels_list[i]  # (T,)

            # Apply mean normalization (matching EEND: logmelmeannorm)
            mix_feat = mix_feat - mix_feat.mean(dim=0, keepdim=True)
            enr_feat = enr_feat - enr_feat.mean(dim=0, keepdim=True)

            # Apply splicing (context concatenation)
            if self.context_size > 0:
                mix_feat = splice(mix_feat, self.context_size)  # (T, F * (2*context+1))
                enr_feat = splice(enr_feat, self.context_size)  # (T_enroll, F * (2*context+1))

            # Apply subsampling
            if self.subsampling > 1:
                mix_feat, labels = subsample_torch(mix_feat, labels, self.subsampling)
                enr_feat, _ = subsample_torch(enr_feat, 
                                           torch.arange(enr_feat.size(0)), 
                                           self.subsampling)

            processed_features.append(mix_feat)
            processed_enroll_features.append(enr_feat)
            processed_labels.append(labels)

        # Pad sequences to same length
        features = pad_sequence(processed_features, batch_first=True)  # (B, T_max, F_spliced)
        enroll_features = pad_sequence(processed_enroll_features, batch_first=True)  # (B, T_enroll_max, F_spliced)
        labels = pad_sequence(processed_labels, batch_first=True, padding_value=self.IGNORE_INDEX)  # (B, T_max)

        # Compute actual lengths after processing
        features_lens = torch.tensor([feat.size(0) for feat in processed_features], dtype=torch.long)
        enroll_features_lens = torch.tensor([feat.size(0) for feat in processed_enroll_features], dtype=torch.long)

        return {
            "features": features,
            "features_lens": features_lens,
            "enroll_features": enroll_features,
            "enroll_features_lens": enroll_features_lens,
            "labels": labels,
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

        # Randomly select a start position within the segment
        max_start_offset = selected_segment.duration - enroll_len
        start_offset = random.uniform(
            0, max_start_offset) if max_start_offset > 0 else 0

        # Calculate absolute start time
        enroll_start = selected_segment.start + start_offset

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
