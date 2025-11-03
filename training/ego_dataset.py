import random

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Optional, List

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


def subsample(Y: np.ndarray, T: np.ndarray, subsample: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Simple subsampling utility function

    Args:
        Y (np.ndarray): Input features
        T (np.ndarray): Targer speakers (lables)
        subsample (int, optional): the subsampling factor. Defaults to 10.

    Returns:
        tuple[np.ndarray, np.ndarray]: a tuple of subsampled values. 
    """
    Y_ss = Y[::subsample]
    T_ss = T[::subsample]
    return Y_ss, T_ss


class EgoCentricDiarizationDataset(Dataset):
    """
    A PyTorch Dataset for ego-centric diarization (TS-DIA) that
    extracts target speaker enrollment chunks *from the cut itself*.

    For each cut, it generates N new training examples (one per speaker).
    Each example contains:
    1. The features for the *full mixture* (X)
    2. The features for *only the target speaker's speech* (the enrollment data)
    3. The 1D categorical labels (Y) from that speaker's perspective

    Output batch dictionary:
    .. code-block::
        {
            'features': (B x T x F) tensor (the full mixture)
            'features_lens': (B,) tensor
            'enroll_features': (B x T_enroll x F) tensor (target speaker's speech)
            'enroll_features_lens': (B,) tensor
            'labels': (B x T) tensor (the 5-class ego-centric labels)
        }
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
        chunk_duration: Seconds = 0.0625,
        min_enroll_len: float = 1.0,
        max_enroll_len: float = 5.0,
        frame_stack: int = 1,
        subsampling: int = 1,
    ) -> None:
        super().__init__()
        validate(cuts)
        self.min_enroll_len = min_enroll_len
        self.max_enroll_len = max_enroll_len
        self.frame_stack = frame_stack
        self.subsampling = subsampling
        self.cuts = cuts

        self.cuts = self.__cut_into_windows(
            cuts=self.cuts, chunk_duration=chunk_duration)
        self.cuts.describe()

    def __getitem__(self, batch_cuts: CutSet) -> Dict[str, torch.Tensor]:
        """
        Takes a CutSet (a mini-batch of cuts) and returns a collated batch
        of (cut, target_speaker) examples.
        """
        mixture_cuts: List[Cut] = []
        enroll_cuts: List[Cut] = []
        labels_list: List[torch.Tensor] = []

        # 1. Expand the batch
        for cut in batch_cuts:
            # Get unique speakers from supervisions
            all_speakers_in_cut: List[str] = sorted(
                set(s.speaker for s in cut.supervisions if s.speaker)
            )
            if not all_speakers_in_cut:
                continue

            for target_spk_id in all_speakers_in_cut:
                # Filter supervisions for target speaker only (no overlap)
                target_supervisions: List[SupervisionSegment] = [
                    s for s in cut.supervisions if s.speaker == target_spk_id
                ]

                if not target_supervisions:
                    continue

                # Get enrollment segment by randomly sampling from continuous speech
                enroll_cut = self._get_random_enrollment(
                    cut, target_spk_id, target_supervisions
                )

                if enroll_cut is None:
                    continue

                # Add the full mixture cut
                mixture_cuts.append(cut)

                # Add the corresponding enrollment cut
                enroll_cuts.append(enroll_cut)

                # Generate and add the categorical labels
                labels_list.append(
                    self._generate_ego_centric_labels(cut, target_spk_id)
                )

        if not mixture_cuts:
            # Batch was empty or no valid speakers were found
            return {
                "features": torch.empty(0, 0, 0),
                "features_lens": torch.empty(0),
                "enroll_features": torch.empty(0, 0, 0),
                "enroll_features_lens": torch.empty(0),
                "labels": torch.empty(0, 0),
            }

        # 2. Collate the expanded batch

        # Collate mixture features: (B x T x F)
        features, features_lens = collate_features(
            CutSet.from_cuts(mixture_cuts)
        )

        # Apply mean normalization per utterance (matching EEND: logmelmeannorm)
        # Subtract mean feature vector from each utterance
        for i in range(features.size(0)):
            valid_len = features_lens[i]
            mean = features[i, :valid_len, :].mean(dim=0, keepdim=True)
            features[i, :valid_len, :] = features[i, :valid_len, :] - mean

        # Collate enrollment features: (B x T_enroll x F)
        enroll_features, enroll_features_lens = collate_features(
            CutSet.from_cuts(enroll_cuts)
        )

        # Apply mean normalization to enrollment features too
        for i in range(enroll_features.size(0)):
            valid_len = enroll_features_lens[i]
            mean = enroll_features[i, :valid_len, :].mean(dim=0, keepdim=True)
            enroll_features[i, :valid_len,
                            :] = enroll_features[i, :valid_len, :] - mean

        # Apply subsampling if configured (like Kaldi: keep every Nth frame)
        if self.subsampling > 1:
            features = features[:, ::self.subsampling, :]
            enroll_features = enroll_features[:, ::self.subsampling, :]
            features_lens = (features_lens + self.subsampling -
                             1) // self.subsampling  # Ceiling division
            enroll_features_lens = (
                enroll_features_lens + self.subsampling - 1) // self.subsampling
            # Subsample labels too
            labels_list = [label[::self.subsampling] for label in labels_list]

        # Collate labels: (B x T)
        labels = pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=self.IGNORE_INDEX
        )

        return {
            "features": features,
            "features_lens": features_lens,
            "enroll_features": enroll_features,
            "enroll_features_lens": enroll_features_lens,
            "labels": labels,
        }

    def __cut_into_windows(self, cuts: CutSet, chunk_duration: Seconds, keep_excessive_supervisions: bool = False) -> CutSet:
        print("cutting the cuts into window sized pieces...")
        new_cuts: CutSet = cuts.cut_into_windows(
            duration=chunk_duration, keep_excessive_supervisions=keep_excessive_supervisions)
        return new_cuts

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
        if num_frames == 0:
            return torch.empty(0, dtype=torch.long)

        frame_shift = cut.frame_shift

        target_mask = torch.zeros(num_frames, dtype=torch.bool)
        other_speaker_count = torch.zeros(num_frames, dtype=torch.int)

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

    def _supervision_to_frames(self, sup: SupervisionSet, num_frames: int, frame_shift: float):
        """Helper to convert supervision start/end times to frame indices."""
        start_frame = round(sup.start / frame_shift)
        end_frame = start_frame + round(sup.duration / frame_shift)
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        return start_frame, end_frame
