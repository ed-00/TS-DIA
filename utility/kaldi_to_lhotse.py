
"""
Utility module for converting Kaldi data directories to Lhotse format.

This module provides functionality to convert simulated audio data from Kaldi format
to Lhotse format, preserving the original directory structure and metadata.
The conversion process extracts speaker, beta, and mixture parameters from directory
names and creates corresponding Lhotse recording and supervision sets.
"""

from pathlib import Path
from typing import Any, Dict, Optional, List
from utility.kaldi_patch import load_kaldi_data_dir
from tqdm import tqdm
import argparse

from lhotse.utils import (
    Pathlike,
    Seconds
)
import re


def __extract_num(s: str) -> int | None:
    """
    Extract the first integer found in a string.
    
    Args:
        s (str): Input string to search for integers
        
    Returns:
        int | None: First integer found in the string, or None if no integer is found
        
    Example:
        >>> __extract_num("ns3_beta7")
        3
        >>> __extract_num("no_numbers")
        None
    """
    m = re.search(r'\d+', s)
    return int(m.group()) if m else None


def __get_simulated_data_parameters(dir_name: str) -> Dict[str, Any]:
    """
    Parse simulated data parameters from directory name.
    
    Extracts dataset split, number of speakers, beta values, and mixture count
    from a standardized directory naming convention.
    
    Args:
        dir_name (str): Directory name following the format:
                       "{split}_clean_5_ns{num_spk}_beta{betas}_{num_mix}"
                       Example: "train_clean_5_ns3_beta7_500"
    
    Returns:
        Dict[str, Any]: Dictionary containing parsed parameters:
            - split (str): Dataset split (e.g., "train", "dev", "test")
            - num_spk (int): Number of speakers
            - betas (int): Beta parameter value
            - num_mix (int): Number of mixtures
    
    Raises:
        ValueError: If directory name doesn't follow expected format or
                   required numeric parameters cannot be extracted
                   
    Example:
        >>> __get_simulated_data_parameters("train_clean_5_ns3_beta7_500")
        {'split': 'train', 'num_spk': 3, 'betas': 7, 'num_mix': 500}
    """
    params = dir_name.split("_")
    split = params[0]
    num_spk = __extract_num(params[3])  # Extract from "ns{num_spk}" format

    betas = __extract_num(params[4])    # Extract from "beta{betas}" format
    num_mix = __extract_num(params[5])  # Extract mixture count

    if num_mix is None or num_spk is None or betas is None:
        raise ValueError(
            "Incorrectly formatted directory names. Expected format: "
            "'{split}_clean_5_ns{num_spk}_beta{betas}_{num_mix}' "
            "Example: 'train_clean_5_ns3_beta7_500'")

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
    """
    Convert Kaldi data directories to Lhotse format.
    
    Processes multiple Kaldi data directories containing simulated speech data,
    converts them to Lhotse RecordingSet and SupervisionSet objects, and saves
    them as compressed JSONL files. The function preserves the original metadata
    by extracting parameters from directory names and incorporating them into
    output filenames.
    
    Args:
        simu_dir (Pathlike): Path to the root directory containing Kaldi data directories.
                            Each subdirectory should follow the naming convention:
                            "{split}_clean_5_ns{num_spk}_beta{betas}_{num_mix}"
        sampling_rate (int): Audio sampling rate in Hz (e.g., 8000, 16000)
        frame_shift (Optional[Seconds], optional): Frame shift for feature extraction.
                                                  Defaults to None.
        map_string_to_underscores (Optional[str], optional): String replacement pattern
                                                           for ID mapping. Defaults to None.
        use_reco2dur (bool, optional): Whether to use reco2dur file for duration info.
                                      Defaults to True.
        num_jobs (int, optional): Number of parallel jobs for processing. Defaults to 1.
        feature_type (str, optional): Type of features to extract. Defaults to "kaldi-fbank".
        output_dir (Pathlike, optional): Base directory for saving converted data.
                                       Defaults to "/workspace/outputs/manifests".
    
    Returns:
        List[Pathlike]: List of unique output directory paths where converted data was saved.
                       Each path corresponds to a different speaker configuration.
    
    Raises:
        AssertionError: If the specified simu_dir doesn't exist
        ValueError: If directory names don't follow expected format (raised by
                   __get_simulated_data_parameters)
    
    Example:
        >>> output_paths = convert_kaldi_to_lhotse(
        ...     simu_dir="/path/to/kaldi/data",
        ...     sampling_rate=16000,
        ...     output_dir="/workspace/outputs/manifests"
        ... )
        >>> print(f"Converted data saved to {len(output_paths)} directories")
        
    Note:
        - Creates directory structure: {output_dir}/simu_{num_spk}spk/
        - Saves files with names indicating parameters:
          * {dataset_name}_recordings_{split}_b{betas}_mix{num_mix}.jsonl.gz
          * {dataset_name}_supervisions_{split}_b{betas}_mix{num_mix}.jsonl.gz
        - Warns if supervision sets are missing from input data
    """
    kaldi_dir: Path = Path(simu_dir)
    assert kaldi_dir.exists(), (
        f"Data directory does not exist: Ensure that {simu_dir} is the correct path"
    )

    # Get all subdirectory names from the Kaldi data directory
    paths: List[str] = [
        el.name for el in kaldi_dir.iterdir() if el.is_dir()]

    # Track unique output directories to avoid duplicates
    saved_set = set()

    # Process each Kaldi data directory
    for data_dir in tqdm(paths, desc="Converting Kaldi data to Lhotse format"):
        # Parse directory name to extract simulation parameters
        params = __get_simulated_data_parameters(data_dir)
        
        # Load Kaldi data using the patched loader
        recording_set, supervision_set, _ = load_kaldi_data_dir(
            path=kaldi_dir / data_dir,
            sampling_rate=sampling_rate,
            frame_shift=frame_shift,
            map_string_to_underscores=map_string_to_underscores,
            use_reco2dur=use_reco2dur,
            num_jobs=num_jobs,
            feature_type=feature_type
        )
        
        # Create dataset name based on number of speakers
        dataset_name = f"simu_{params['num_spk']}spk"
        output_path = Path(output_dir) / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate output filenames with parameter information
        recording_path = (
            output_path / 
            f"{dataset_name}_recordings_{params['split']}_b{params['betas']}_mix{params['num_mix']}.jsonl.gz"
        )
        supervision_path = (
            output_path / 
            f"{dataset_name}_supervisions_{params['split']}_b{params['betas']}_mix{params['num_mix']}.jsonl.gz"
        )

        # Save recording set (always present)
        recording_set.to_file(recording_path)
        saved_set.add(output_path)
        
        # Save supervision set if available
        if supervision_set:
            supervision_set.to_file(supervision_path)
        else:
            print(f"⚠️ Warning: supervision set does not exist for {data_dir}")

    return list(saved_set)


def main():
    """Main function with argument parsing for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert Kaldi data directories to Lhotse format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "simu_dir",
        type=str,
        help="Path to the root directory containing Kaldi data directories"
    )
    parser.add_argument(
        "--sampling-rate", "-sr",
        type=int,
        required=True,
        help="Audio sampling rate in Hz (e.g., 8000, 16000)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--frame-shift", "-fs",
        type=float,
        default=None,
        help="Frame shift for feature extraction in seconds"
    )
    parser.add_argument(
        "--map-string-to-underscores",
        type=str,
        default=None,
        help="String replacement pattern for ID mapping"
    )
    parser.add_argument(
        "--use-reco2dur",
        action="store_true",
        default=True,
        help="Use reco2dur file for duration info (default: True)"
    )
    parser.add_argument(
        "--no-reco2dur",
        dest="use_reco2dur",
        action="store_false",
        help="Don't use reco2dur file for duration info"
    )
    parser.add_argument(
        "--num-jobs", "-j",
        type=int,
        default=1,
        help="Number of parallel jobs for processing"
    )
    parser.add_argument(
        "--feature-type", "-ft",
        type=str,
        default="kaldi-fbank",
        choices=["kaldi-fbank", "kaldi-mfcc", "kaldi-plp"],
        help="Type of features to extract"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="/workspace/outputs/manifests",
        help="Base directory for saving converted data"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not Path(args.simu_dir).exists():
        parser.error(f"Input directory does not exist: {args.simu_dir}")
    
    if args.verbose:
        print(f"Converting Kaldi data from: {args.simu_dir}")
        print(f"Sampling rate: {args.sampling_rate} Hz")
        print(f"Output directory: {args.output_dir}")
        print(f"Feature type: {args.feature_type}")
        print(f"Number of jobs: {args.num_jobs}")
    
    # Convert Kaldi data to Lhotse format
    try:
        output_paths = convert_kaldi_to_lhotse(
            simu_dir=args.simu_dir,
            sampling_rate=args.sampling_rate,
            frame_shift=args.frame_shift,
            map_string_to_underscores=args.map_string_to_underscores,
            use_reco2dur=args.use_reco2dur,
            num_jobs=args.num_jobs,
            feature_type=args.feature_type,
            output_dir=args.output_dir
        )
        
        print(f"Conversion completed successfully!")
        print(f"Data saved to {len(output_paths)} directories:")
        for path in sorted(output_paths):
            print(f"   • {path}")
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
