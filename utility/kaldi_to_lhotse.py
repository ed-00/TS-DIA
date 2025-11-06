
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from utility.kaldi_patch import load_kaldi_data_dir
from tqdm import tqdm

from lhotse.utils import (
    Pathlike,
    Seconds
)
import re


def __extract_num(s: str) -> int | None:
    m = re.search(r'\d+', s)
    return int(m.group()) if m else None


def __get_simulated_data_parameters(dir_name: str) -> Dict[str, Any]:
    params = dir_name.split("_")
    split = params[0]
    num_spk = __extract_num(params[3])

    betas = __extract_num(params[4])
    num_mix = __extract_num(params[5])

    if num_mix is None or num_spk is None or betas is None:
        raise ValueError(
            "Incorrectly formated directory names, the directory names should be formated like this: train_clean_5_ns3_beta7_500")

    return {
        "split": split,
        "betas": betas,
        "num_mix": num_mix,
        "num_spk": num_spk
    }


def convert_kaldi_to_lhotse(
    simu_dir: Pathlike,
    sampling_rate: int,
    frame_shift: Optional[Seconds] = None,
    map_string_to_underscores: Optional[str] = None,
    use_reco2dur: bool = True,
    num_jobs: int = 1,
    feature_type: str = "kaldi-fbank",
    output_dir: Pathlike = "/workspace/outputs/manifests"
) -> List[Pathlike]:
    kaldi_dir: Path = Path(simu_dir)
    assert kaldi_dir.exists(
    ), f"Data dirctory dose not exit: Ensure that {simu_dir} is the correct path"

    paths: List[str] = [
        el.name for el in kaldi_dir.iterdir() if el.is_dir()]

    saved_set = set()

    for data_dir in tqdm(paths, desc="loading kaldi data"):
        params = __get_simulated_data_parameters(
            data_dir)
        recording_set, supervision_set, _ = load_kaldi_data_dir(
            path=kaldi_dir / data_dir,
            sampling_rate=sampling_rate,
            frame_shift=frame_shift,
            map_string_to_underscores=map_string_to_underscores,
            use_reco2dur=use_reco2dur,
            num_jobs=num_jobs,
            feature_type=feature_type
        )
        dataset_name = f"simu_{params['num_spk']}spk"
        output_path = f"{output_dir}/{dataset_name}"
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        recording_path = f"{output_path}/{dataset_name}_recordings_{params['split']}_b{params['betas']}_mix{params['num_mix']}.jsonl.gz"
        supervision_path = f"{output_path}/{dataset_name}_supervisions_{params['split']}_b{params['betas']}_mix{params['num_mix']}.jsonl.gz"

        recording_set.to_file(recording_path)
        saved_set.add(output_path)
        if supervision_set:
            supervision_set.to_file(supervision_path)
        else:
            print(f"⚠️ Warning: supervision set dose not exit for {data_dir} ")

    return list(saved_set)


if __name__ == "__main__":
    convert_kaldi_to_lhotse(
        simu_dir="/workspace/data_simu/mini_librispeech/data/simu/data", sampling_rate=8000)
