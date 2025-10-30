import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Optional, List

from lhotse import validate, CutSet
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_features
from lhotse.supervision import SupervisionSet, RecordingSet


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
        'others_sgl': 2, # exactly one non-target speaker
        'others_ovl': 3, # overlap of non-target speakers
        'ns': 4          # non-speech
    }
    IGNORE_INDEX = CrossEntropyLoss().ignore_index

    def __init__(
        self,
        cuts: CutSet,
        uem: Optional[SupervisionSet] = None,
    ) -> None:
        super().__init__()
        validate(cuts)
        if not uem:
            self.cuts = cuts
        else:
            # We use the `overlap` method in intervaltree to get overlapping regions
            # between the supervision segments and the UEM segments
            recordings = RecordingSet(
                {c.recording.id: c.recording for c in cuts if c.has_recording}
            )
            uem_intervals = CutSet.from_manifests(
                recordings=recordings,
                supervisions=uem,
            ).index_supervisions()
            supervisions = []
            for cut_id, tree in cuts.index_supervisions().items():
                if cut_id not in uem_intervals:
                    supervisions += [it.data for it in tree]
                    continue
                supervisions += {
                    it.data.trim(it.end, start=it.begin)
                    for uem_it in uem_intervals[cut_id]
                    for it in tree.overlap(begin=uem_it.begin, end=uem_it.end)
                }
            self.cuts = CutSet.from_manifests(
                recordings=recordings,
                supervisions=SupervisionSet.from_segments(supervisions),
            )
        self.cuts = cuts

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
            all_speakers_in_cut = cut.speakers
            if not all_speakers_in_cut:
                continue

            for target_spk_id in all_speakers_in_cut:
                # Get the enrollment speech from this cut
                # using the ground-truth supervisions.
                enroll_cutset = cut.trim_to_supervisions(
                    speaker=target_spk_id,
                    keep_overlapping=False
                )

                # If the speaker has no speech in this cut, we can't enroll.
                if not enroll_cutset:
                    continue

                # Stack all their speech segments into one single cut
                enroll_cut = enroll_cutset.stack()

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

        # Collate enrollment features: (B x T_enroll x F)
        enroll_features, enroll_features_lens = collate_features(
            CutSet.from_cuts(enroll_cuts)
        )

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

        other_speaker_ids = {
            spk for spk in cut.speakers if spk != target_speaker_id
        }

        # Create target speaker mask
        for sup in cut.supervisions.filter(lambda s: s.speaker == target_speaker_id):
            start_frame, end_frame = self._supervision_to_frames(sup, num_frames, frame_shift)
            target_mask[start_frame:end_frame] = True

        # Create other speaker masks and sum them
        for other_spk_id in other_speaker_ids:
            other_spk_mask = torch.zeros(num_frames, dtype=torch.bool)
            for sup in cut.supervisions.filter(lambda s: s.speaker == other_spk_id):
                start_frame, end_frame = self._supervision_to_frames(sup, num_frames, frame_shift)
                other_spk_mask[start_frame:end_frame] = True
            other_speaker_count += other_spk_mask.int()

        # Map masks to the 5 categorical labels
        labels = torch.full(
            (num_frames,),
            fill_value=self.LABEL_MAP['ns'],
            dtype=torch.long
        )
        labels[(target_mask == True) & (other_speaker_count == 0)] = self.LABEL_MAP['ts']
        labels[(target_mask == True) & (other_speaker_count > 0)] = self.LABEL_MAP['ts_ovl']
        labels[(target_mask == False) & (other_speaker_count == 1)] = self.LABEL_MAP['others_sgl']
        labels[(target_mask == False) & (other_speaker_count > 1)] = self.LABEL_MAP['others_ovl']

        return labels

    def _supervision_to_frames(self, sup: SupervisionSet, num_frames: int, frame_shift: float):
        """Helper to convert supervision start/end times to frame indices."""
        start_frame = round(sup.start / frame_shift)
        end_frame = start_frame + round(sup.duration / frame_shift)
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        return start_frame, end_frame