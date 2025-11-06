from typing import List, Literal, Optional
from lhotse.utils import Pathlike
import random
from lhotse import CutSet

RATIO_TYPE = Literal['count_ratio', 'count_based', 'duration_based']
HOURS = float


def __take_by_duration(cuts: CutSet, duration: HOURS, shuffle: bool, random_seed: Optional[random.Random]) -> CutSet:
    if shuffle:
        cuts = cuts.shuffle(rng=random_seed)


def __take_by_count_ratio(cuts: CutSet, cut_ratio: float, shuffle: bool, random_seed: Optional[random.Random]) -> CutSet:
    if shuffle:
        cuts = cuts.shuffle(random_seed)
    n_cuts = int(1 - cut_ratio * len(cuts))
    return cuts.subset(first=n_cuts)





def combine_data(
    cuts: List[CutSet],
    ratio_mapping: Optional[List[float | int]],
    shuffle: bool = True,
    random_seed: int = 69,
    output_path: Pathlike = '/workspace/manifests/simu_combo'
):
    """Utility function to combine multipule cutsets.

    Returns:
        CutSet: a combined cutset. 
    """
