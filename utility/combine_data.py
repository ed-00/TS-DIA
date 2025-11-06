"""
Utility module for combining multiple Lhotse CutSet datasets.

This module provides functionality to combine multiple CutSet objects with optional
subsetting based on count ratios or absolute counts. It supports shuffling datasets
before combination and can save the combined result to disk.
"""

from lhotse.manipulation import combine
from typing import List, Literal, Optional
from lhotse.utils import Pathlike
from lhotse import CutSet
from pathlib import Path
import random
import argparse

RatioType = Literal['count_ratio', 'count_based']
Hours = float
HOURS_TO_SEC = 3600.0


def __take_by_count_ratio(cuts: CutSet, cut_ratio: float) -> CutSet:
    """
    Take a subset of cuts based on a ratio of the total count.
    
    Args:
        cuts (CutSet): Input CutSet to subsample
        cut_ratio (float): Ratio of cuts to keep (0.0 to 1.0)
        
    Returns:
        CutSet: Subset of the original CutSet
    """
    num_cuts = int((1 - cut_ratio) * len(cuts))
    return cuts.subset(first=num_cuts)


def __shuffle_all(cuts: list[CutSet], random_seed: Optional[random.Random]) -> List[CutSet]:
    """
    Shuffle all CutSets in a list using the same random seed.
    
    Args:
        cuts (list[CutSet]): List of CutSets to shuffle
        random_seed (Optional[random.Random]): Random number generator for reproducibility
        
    Returns:
        List[CutSet]: List of shuffled CutSets
    """
    return [c.shuffle(rng=random_seed) for c in cuts]


def __take_by_count(cuts: CutSet, num_cuts: int) -> CutSet:
    """
    Take a subset of cuts based on absolute count.
    
    Args:
        cuts (CutSet): Input CutSet to subsample
        num_cuts (int): Number of cuts to keep
        
    Returns:
        CutSet: Subset of the original CutSet
    """
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
    """
    Combine multiple CutSet datasets with optional subsetting and shuffling.
    
    This function takes a list of CutSet objects, optionally applies subsetting
    based on ratios or absolute counts, shuffles them if requested, and combines
    them into a single CutSet. The result can be saved to disk.
    
    Args:
        cuts (List[CutSet]): List of CutSets to combine
        shuffle (bool, optional): Whether to shuffle each dataset before combining.
                                Defaults to True.
        random_seed (Optional[random.Random], optional): Random number generator
                                                        for deterministic output.
                                                        Defaults to None.
        ratio_type (Optional[RatioType], optional): Type of subsetting to apply.
                                                  Either 'count_ratio' or 'count_based'.
                                                  Defaults to None.
        subset_mapping (Optional[List[float | int]], optional): List of values for
                                                               subsetting each CutSet.
                                                               Must match length of cuts list.
                                                               Defaults to None.
        output_path (Optional[Pathlike], optional): Directory path where the combined
                                                  dataset will be saved.
                                                  Defaults to '/workspace/manifests/simu_combo'.
        file_name (str, optional): Name of the output file.
                                 Defaults to 'simu_combo_supervisions_dev.jsonl.gz'.
    
    Returns:
        CutSet: Combined CutSet containing all input datasets
        
    Raises:
        AssertionError: If subset_mapping length doesn't match cuts length
        
    Example:
        >>> cuts = [cutset1, cutset2, cutset3]
        >>> combined = combine_data(
        ...     cuts=cuts,
        ...     shuffle=True,
        ...     ratio_type='count_based',
        ...     subset_mapping=[100, 200, 150],
        ...     output_path='/output/dir',
        ...     file_name='combined_data.jsonl.gz'
        ... )
    """
    # Shuffle all CutSets if requested
    if shuffle:
        cuts = __shuffle_all(cuts=cuts, random_seed=random_seed)
        
    # Apply subsetting if parameters are provided
    if subset_mapping is not None and ratio_type is not None:
        modified_cuts: List[CutSet] = []
        assert len(cuts) == len(subset_mapping), (
            "The subset mapping must be the same length as the cutset length"
        )
        
        for cutset, mapping in zip(cuts, subset_mapping):
            if ratio_type == 'count_ratio':
                processed_cutset: CutSet = __take_by_count_ratio(
                    cuts=cutset, cut_ratio=float(mapping))
            else:  # count_based
                processed_cutset: CutSet = __take_by_count(
                    cuts=cutset, num_cuts=int(mapping))
            modified_cuts.append(processed_cutset)

        combined_cuts = combine(modified_cuts)
    else:
        # Combine all CutSets without subsetting
        combined_cuts = combine(cuts)

    # Save to file if output path is specified
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        combined_cuts.to_file(output_path / file_name)

    return combined_cuts


def load_cutsets_from_files(file_paths: List[Pathlike]) -> List[CutSet]:
    """
    Load CutSets from a list of .jsonl.gz file paths.
    
    Args:
        file_paths (List[Pathlike]): List of file paths to .jsonl.gz files
                                   containing CutSet data
    
    Returns:
        List[CutSet]: List of loaded CutSets
        
    Raises:
        FileNotFoundError: If any of the specified files doesn't exist
        ValueError: If files cannot be loaded as CutSets
        
    Example:
        >>> paths = [
        ...     '/path/to/dataset1_supervisions.jsonl.gz',
        ...     '/path/to/dataset2_supervisions.jsonl.gz'
        ... ]
        >>> cutsets = load_cutsets_from_files(paths)
    """
    cutsets = []
    for file_path in file_paths:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        try:
            cutset = CutSet.from_file(file_path)
            cutsets.append(cutset)
            print(f"Loaded {len(cutset)} cuts from {file_path}")
        except Exception as e:
            raise ValueError(f"Failed to load CutSet from {file_path}: {e}")
    
    return cutsets


def main():
    """Main function with argument parsing for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Combine multiple Lhotse CutSet datasets from .jsonl.gz files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "input_files",
        nargs="+",
        help="List of .jsonl.gz files containing CutSet data (supervisions or recordings)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="/workspace/outputs/manifests",
        help="Output directory for combined dataset"
    )
    parser.add_argument(
        "--output-name", "-n",
        type=str,
        default="combined_dataset.jsonl.gz",
        help="Output file name"
    )
    
    # Processing options
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle datasets before combining"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducible shuffling"
    )
    
    # Subsetting options
    parser.add_argument(
        "--ratio-type",
        choices=["count_ratio", "count_based"],
        default=None,
        help="Type of subsetting to apply"
    )
    parser.add_argument(
        "--subset-mapping",
        nargs="+",
        type=float,
        default=None,
        help="Subsetting values for each input file (ratios or counts)"
    )
    
    # Other options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    for file_path in args.input_files:
        if not Path(file_path).exists():
            parser.error(f"Input file does not exist: {file_path}")
        if not file_path.endswith(('.jsonl.gz', '.jsonl')):
            parser.error(f"Input file must be .jsonl.gz or .jsonl: {file_path}")
    
    if args.subset_mapping and not args.ratio_type:
        parser.error("--ratio-type must be specified when using --subset-mapping")
    
    if args.subset_mapping and len(args.subset_mapping) != len(args.input_files):
        parser.error("Number of subset mapping values must match number of input files")
    
    # Setup random seed
    random_seed = None
    if args.random_seed is not None:
        random_seed = random.Random(args.random_seed)
    
    if args.verbose:
        print(f"Loading {len(args.input_files)} CutSet files...")
        print(f"Output directory: {args.output_dir}")
        print(f"Output file: {args.output_name}")
        print(f"Shuffle: {not args.no_shuffle}")
        if args.ratio_type:
            print(f"Subsetting type: {args.ratio_type}")
            print(f"Subset mapping: {args.subset_mapping}")
    
    try:
        # Load CutSets from files
        cutsets = load_cutsets_from_files(args.input_files)
        
        if args.verbose:
            total_cuts = sum(len(cs) for cs in cutsets)
            print(f"Total cuts loaded: {total_cuts}")
        
        # Combine datasets
        combined_cutset = combine_data(
            cuts=cutsets,
            shuffle=not args.no_shuffle,
            random_seed=random_seed,
            ratio_type=args.ratio_type,
            subset_mapping=args.subset_mapping,
            output_path=args.output_dir,
            file_name=args.output_name
        )
        
        print(f"Successfully combined datasets!")
        print(f"Combined dataset contains {len(combined_cutset)} cuts")
        print(f"Output saved to: {Path(args.output_dir) / args.output_name}")
        
    except Exception as e:
        print(f"Error during combination: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
