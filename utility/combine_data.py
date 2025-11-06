from lhotse.manipulation import combine
from typing import List, Literal, Optional
from lhotse.utils import Pathlike
from lhotse import CutSet
from pathlib import Path
import random

RatioType = Literal['count_ratio', 'count_based']
Hours = float
HOURS_TO_SEC = 3600.0


def __take_by_count_ratio(cuts: CutSet, cut_ratio: float) -> CutSet:

    num_cuts = int(1 - cut_ratio * len(cuts))
    return cuts.subset(first=num_cuts)


def __shuffle_all(cuts: list[CutSet], random_seed: Optional[random.Random]) -> List[CutSet]:
    return [c.shuffle(rng=random_seed) for c in cuts]


def __take_by_count(cuts: CutSet, num_cuts: int) -> CutSet:
    return cuts.subset(first=num_cuts)


def combine_data(
    cuts: List[CutSet],
    shuffle: bool = True,
    random_seed: Optional[random.Random] = None,
    ratio_type: Optional[RatioType] = None,
    subset_mapping: Optional[List[float | int]] = None,
    output_path: Optional[Pathlike] = '/workspace/manifests/simu_combo',
    file_name: str = 'simu_combo_supervisions_dev.jsonl.gz'
) -> CutSet:
    """Utility function for combaining datasets 

    Args:
        cuts (List[CutSet]): list of cutset to combined 
        shuffle (bool, optional): weather to suffle the dataset or not. Defaults to True.
        random_seed (Optional[random.Random], optional): random object for determanistic outpu. Defaults to None.
        subset_mapping (Optional[List[float  |  int]], optional): list of float or int incdication count. Defaults to None.
        ratio_type (Optional[RatioType], optional): count_base or ratio_based. Defaults to None.
        output_path (Optional[Pathlike], optional): save path of the combined dataset. Defaults to '/workspace/manifests/simu_combo'.
        file_name (str, optional): The name of the saved file. Defaults to 'simu_combo_supervisions_dev.jsonl.gz'.

    Returns:
        CutSet: _description_
    """
    if shuffle:
        cuts = __shuffle_all(cuts=cuts, random_seed=random_seed)
    if subset_mapping is not None and ratio_type is not None:
        modified_cuts: List[CutSet] = []
        assert len(cuts) == len(
            subset_mapping), "The subset mapping must be the same length as the cutset length"
        for cutset, mapping, in zip(cuts, subset_mapping):
            if subset_mapping == 'count_ratio':
                processed_cutset: CutSet = __take_by_count_ratio(
                    cuts=cutset, cut_ratio=float(mapping))
            else:
                processed_cutset: CutSet = __take_by_count(
                    cuts=cutset, num_cuts=int(mapping))
            modified_cuts.append(processed_cutset)

        combined_cuts = combine(modified_cuts)
    else:
        combined_cuts = combine(cuts)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        combined_cuts.to_file(output_path / file_name)

    return combined_cuts
