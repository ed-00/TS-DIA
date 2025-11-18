import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data import IterableDataset, get_worker_info
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Optional, List, Tuple, cast
import numpy as np
from tqdm.auto import tqdm

from lhotse import CutSet
from lhotse.lazy import LazySlicer
from lhotse.cut import MonoCut, Cut

Seconds = float


def splice(Y: torch.Tensor, context_size: int = 7) -> torch.Tensor:
    """
    Feature splicing (concatenate adjacent frames).
    
    :param Y: Input features tensor of shape (T, F).
    :param context_size: Number of frames to stack on left and right.
    :return: Spliced features tensor of shape (T, F * (2 * context_size + 1)).
    """
    T, F = Y.shape
    total_context_frames = 2 * context_size + 1
    spliced_dim = F * total_context_frames
    
    # Pre-allocate tensor for spliced features
    spliced_Y = torch.zeros(T, spliced_dim, dtype=Y.dtype, device=Y.device)
    
    # Pad Y on the time axis for easier context gathering
    # Pad with first frame on the left, last frame on the right
    padded_Y = torch.cat(
        [Y[0].unsqueeze(0).repeat(context_size, 1)] +
        [Y] +
        [Y[-1].unsqueeze(0).repeat(context_size, 1)],
        dim=0
    )

    # Gather context frames efficiently
    for t in range(T):
        # The true index in padded_Y for the current time t is (t + context_size)
        # We want the window from (t) to (t + 2*context_size) in padded_Y
        context = padded_Y[t : t + total_context_frames]
        spliced_Y[t] = context.view(-1) # Flatten (total_context_frames, F) -> (total_context_frames * F)
        
    return spliced_Y


def subsample_torch(Y: torch.Tensor, T: torch.Tensor, subsample: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Subsampling utility function for PyTorch tensors."""
    Y_ss = Y[::subsample]
    T_ss = T[::subsample]
    return Y_ss, T_ss


class EgoCentricDiarizationDataset(IterableDataset):
    """
    An IterableDataset for Ego-Centric Diarization that streams data 
    from a lazy Lhotse CutSet.
    
    This dataset is designed for:
    -   **Low Memory:** It streams from a lazy CutSet, never loading the
        full manifest into memory.
    -   **Fast Startup:** It avoids building a large index map at init time.
    -   **Multi-Worker:** It correctly partitions the data across DataLoader workers.
    -   **Multi-GPU (Accelerate/DDP):** It correctly partitions data across
        distributed processes (ranks).
    -   **1-to-N Expansion:** It yields one example for each speaker (plus 'None')
        for every cut in the stream.
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
        shuffle_buffer_size: int = 10000
    ) -> None:
        super().__init__()
        self.cuts = cuts
        self.context_size = context_size
        self.subsampling = subsampling
        self.shuffle_buffer_size = shuffle_buffer_size

    @staticmethod
    def get_total_dataset_size(cuts: CutSet, desc="Calculating total dataset size") -> int:
        """
        Scans a CutSet once (lazily) to count the total number of examples
        (sum of (speakers_per_cut + 1) for all cuts) that this dataset
        will yield in one epoch.
        
        :param cuts: The CutSet to scan.
        :param desc: Description for the tqdm progress bar.
        :return: The total number of examples.
        """
        total = 0
        # This loop is lazy and memory-efficient
        for cut in tqdm(cuts, desc=desc, leave=False):
            num_speakers = len(set(s.speaker for s in cut.supervisions if s.speaker))
            total += (num_speakers + 1) # +1 for the 'None' speaker
        return total

    def __iter__(self):
        """
        Yields examples.
        
        Handles partitioning for both distributed training (DDP/Accelerate)
        and multi-worker data loading.
        """
        
        # 1. Determine Distributed (DDP/Accelerate) context
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # 2. Determine DataLoader worker context
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # 3. Calculate global partitioning
        # We have (world_size * num_workers) total consumers of data.
        # Each one needs a unique, non-overlapping slice of the CutSet.
        total_partitions = world_size * num_workers
        current_partition_id = (rank * num_workers) + worker_id

        # 4. Create the partitioned, lazy CutSet *iterable*
        # LazySlicer streams every N-th cut, starting from k-th.
        my_cuts_iterable: CutSet = CutSet(LazySlicer(
            self.cuts, 
            k=current_partition_id, 
            n=total_partitions
        ))

        # 5. Apply approximate shuffling using a buffer
        if self.shuffle_buffer_size > 0:
            my_cuts_iterable = my_cuts_iterable.shuffle(buffer_size=self.shuffle_buffer_size)

        # 6. Start the streaming and processing loop
        for cut in iter(my_cuts_iterable):
            cut = cast(MonoCut, cut)
            
            # Load features once per cut
            try:
                mixture_features = cut.load_features()
                if mixture_features is None:
                    print(f"[Rank {rank}/W {worker_id}] Warning: Failed to load features for cut {cut.id}. Skipping.")
                    continue
            except Exception as e:
                print(f"[Rank {rank}/W {worker_id}] Error loading features for cut {cut.id}: {e}. Skipping.")
                continue

            mixture_features = torch.from_numpy(mixture_features).float()
            mixture_features = mixture_features - mixture_features.mean(dim=0, keepdim=True)

            # Find all target speakers for this cut
            all_speakers_in_cut = sorted(
                set(s.speaker for s in cut.supervisions if s.speaker)
            )
            target_speakers = all_speakers_in_cut + [None]
            
            # Generate all labels for this cut at once
            labels_dict = self.generate_labels_for_cut(
                cut, target_speakers, self.LABEL_MAP
            )

            # Yield one example for each target speaker
            for target_spk_id in target_speakers:
                speaker_key = target_spk_id if target_spk_id else "__none__"
                
                np_labels = labels_dict[speaker_key]
                # Keep original multi-class labels under `labels` (EGO class indices)
                labels = torch.from_numpy(np_labels).long()

                # One-hot labels are unnecessary and were removed to reduce memory
                
                # Clone features to avoid in-place modification bugs
                curr_features = mixture_features.clone()
                
                if self.subsampling > 1:
                    curr_features, labels = subsample_torch(
                        curr_features, labels, subsample=self.subsampling
                    )

                curr_features = splice(
                    curr_features, context_size=self.context_size
                )

                # 0/1 indicator whether this example corresponds to a target speaker or the special __none__
                is_target_flag = 0 if (target_spk_id is None) else 1

                yield {
                    "features": curr_features,
                    # `labels` are multi-class ego labels (0..4): [ts, ts_ovl, others_sgl, others_ovl, ns]
                    "labels": labels,
                    # Keep a per-example binary indicator `is_target` for convenience (0/1)
                    "is_target": is_target_flag,
                }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        """
        Custom collate function to pad features and labels.
        """
        features_list = [item["features"] for item in batch]
        labels_list = [item["labels"] for item in batch]
        # Only keep features/labels/is_target to minimize memory use
        is_target_list = [item["is_target"] for item in batch]

    
        features_padded = pad_sequence(
            features_list, batch_first=True, padding_value=0.0)
        labels_padded = pad_sequence(
            labels_list, batch_first=True, padding_value=EgoCentricDiarizationDataset.IGNORE_INDEX)
        # is_target is provided by dataset as 0/1, so collate it directly as a tensor
        is_target_tensor = torch.tensor(is_target_list, dtype=torch.int64)

        return {
            "features": features_padded,
            "labels": labels_padded,
            "is_target": is_target_tensor,
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

        # Case 1: No speakers at all in the cut
        if not all_speakers_in_cut:
            ns_labels = np.full(
                num_frames, fill_value=label_map['ns'], dtype=np.int64
            )
            for spk_id in speaker_ids:
                speaker_key = spk_id if spk_id else "__none__"
                labels_dict[speaker_key] = ns_labels
            return labels_dict

        # Case 2: Speakers are present
        speaker_to_idx_map = {spk: i for i,
                              spk in enumerate(all_speakers_in_cut)}
        
        # Create the multi-speaker activity mask
        # Shape: (num_speakers, num_frames)
        mask = cut.speakers_feature_mask(speaker_to_idx_map=speaker_to_idx_map)

        # Generate labels for each target speaker
        for target_speaker_id in speaker_ids:
            speaker_key = target_speaker_id if target_speaker_id else "__none__"

            if target_speaker_id is None:
                # Target is "None", so all speakers are "others"
                target_mask = np.zeros(num_frames, dtype=np.int32)
                other_speaker_count = np.sum(mask, axis=0)
            else:
                # Target is a specific speaker
                if target_speaker_id not in speaker_to_idx_map:
                    # This could happen if a speaker ID was passed in
                    # but had no supervisions (and was filtered out)
                    # In this case, treat as "None"
                    target_mask = np.zeros(num_frames, dtype=np.int32)
                    other_speaker_count = np.sum(mask, axis=0)
                else:
                    target_idx = speaker_to_idx_map[target_speaker_id]
                    target_mask = mask[target_idx]
                    
                    # Get activity of all *other* speakers
                    other_indices = [
                        i for i, spk in enumerate(all_speakers_in_cut)
                        if spk != target_speaker_id
                    ]
                    
                    if other_indices:
                        other_speaker_count = np.sum(mask[other_indices], axis=0)
                    else:
                        other_speaker_count = np.zeros(num_frames, dtype=np.int32)

            
            # Start with all non-speech
            labels = np.full(
                num_frames, fill_value=label_map['ns'], dtype=np.int64)
            
            # 'ts': target=1, others=0
            labels[(target_mask == 1) & (other_speaker_count == 0)] = label_map['ts']
            
            # 'ts_ovl': target=1, others>0
            labels[(target_mask == 1) & (other_speaker_count > 0)] = label_map['ts_ovl']
            
            # 'others_sgl': target=0, others=1
            labels[(target_mask == 0) & (other_speaker_count == 1)] = label_map['others_sgl']
            
            # 'others_ovl': target=0, others>1
            labels[(target_mask == 0) & (other_speaker_count > 1)] = label_map['others_ovl']

            labels_dict[speaker_key] = labels

        return labels_dict