"""
Utility module for combining multiple Lhotse CutSet datasets.

This module provides functionality to combine multiple CutSet objects with optional
subsetting based on count ratios or absolute counts. It supports shuffling datasets
before combination and can save the combined result to disk.
"""

from lhotse.manipulation import combine
from typing import List, Optional
from lhotse.utils import Pathlike
from lhotse import CutSet, RecordingSet, SupervisionSet
from pathlib import Path
import argparse

def combine_data(
    cuts: List[CutSet],
    output_path: Optional[Pathlike] = '/workspace/outputs/manifests/combos',
    file_name: str = 'combined_dataset.jsonl.gz'
) -> CutSet:
    """
    Combine multiple CutSet datasets into a single CutSet.
    
    This function takes a list of CutSet objects and combines them into a single
    CutSet. The result can be saved to disk.
    
    Args:
        cuts (List[CutSet]): List of CutSets to combine
        output_path (Optional[Pathlike], optional): Directory path where the combined
                                                  dataset will be saved.
                                                  Defaults to '/workspace/outputs/manifests/combos'.
        file_name (str, optional): Name of the output file.
                                 Defaults to 'combined_dataset.jsonl.gz'.
    
    Returns:
        CutSet: Combined CutSet containing all input datasets
        
    Example:
        >>> cuts = [cutset1, cutset2, cutset3]
        >>> combined = combine_data(
        ...     cuts=cuts,
        ...     output_path='/output/dir',
        ...     file_name='combined_data.jsonl.gz'
        ... )
    """
    # Combine all CutSets
    combined_cuts = combine(cuts)

    # Save to file if output path is specified
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        combined_cuts.to_file(output_path / file_name)
        print(f"Combined dataset saved to: {output_path / file_name}")

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


def load_paired_cutsets_from_manifests(recording_paths: List[Pathlike], supervision_paths: List[Pathlike]) -> List[CutSet]:
    """
    Load CutSets from paired recording and supervision manifest files.
    
    This function ensures that recordings and supervisions are properly aligned
    by loading them together using CutSet.from_manifests(), which maintains
    the correct pairing between audio and labels.
    
    Args:
        recording_paths (List[Pathlike]): List of recording manifest file paths
        supervision_paths (List[Pathlike]): List of supervision manifest file paths
                                          (must correspond 1:1 with recording_paths)
    
    Returns:
        List[CutSet]: List of CutSets with properly paired recordings and supervisions
        
    Raises:
        ValueError: If the number of recording and supervision files don't match
        FileNotFoundError: If any of the specified files doesn't exist
        ValueError: If files cannot be loaded as CutSets
        
    Example:
        >>> rec_paths = [
        ...     '/path/to/dataset1_recordings.jsonl.gz',
        ...     '/path/to/dataset2_recordings.jsonl.gz'
        ... ]
        >>> sup_paths = [
        ...     '/path/to/dataset1_supervisions.jsonl.gz',
        ...     '/path/to/dataset2_supervisions.jsonl.gz'
        ... ]
        >>> cutsets = load_paired_cutsets_from_manifests(rec_paths, sup_paths)
    """
    if len(recording_paths) != len(supervision_paths):
        raise ValueError(
            f"Number of recording files ({len(recording_paths)}) must match "
            f"number of supervision files ({len(supervision_paths)})"
        )
    
    cutsets = []
    for rec_path, sup_path in zip(recording_paths, supervision_paths):
        rec_path = Path(rec_path)
        sup_path = Path(sup_path)
        
        # Validate files exist
        if not rec_path.exists():
            raise FileNotFoundError(f"Recording file does not exist: {rec_path}")
        if not sup_path.exists():
            raise FileNotFoundError(f"Supervision file does not exist: {sup_path}")
        
        try:
            # Load RecordingSet and SupervisionSet first
            recordings = RecordingSet.from_file(rec_path)
            supervisions = SupervisionSet.from_file(sup_path)
            
            # Create paired CutSet to maintain alignment
            cutset = CutSet.from_manifests(
                recordings=recordings,
                supervisions=supervisions
            )
            cutsets.append(cutset)
            print(f"Loaded {len(cutset)} paired cuts from:")
            print(f"  - Recordings: {rec_path}")
            print(f"  - Supervisions: {sup_path}")
        except Exception as e:
            raise ValueError(f"Failed to load paired CutSet from {rec_path} and {sup_path}: {e}")
    
    return cutsets


def main():
    """Main function with argument parsing for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Combine multiple Lhotse CutSet datasets from .jsonl.gz files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input arguments
    parser.add_argument(
        "input_files",
        nargs="*",
        help="List of .jsonl.gz files containing CutSet data (for single file mode)"
    )
    parser.add_argument(
        "--paired-manifests",
        action="store_true",
        help="Use paired recording and supervision manifest mode"
    )
    
    # Paired manifest arguments (only used with --paired-manifests)
    parser.add_argument(
        "--recording-files",
        nargs="+",
        help="List of recording manifest files (required with --paired-manifests)"
    )
    parser.add_argument(
        "--supervision-files",
        nargs="+",
        help="List of supervision manifest files (required with --paired-manifests)"
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
    

    
    # Other options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input mode
    if args.paired_manifests:
        if not args.recording_files or not args.supervision_files:
            parser.error("--recording-files and --supervision-files are required with --paired-manifests")
        
        if len(args.recording_files) != len(args.supervision_files):
            parser.error("Number of recording files must match number of supervision files")
        
        # Validate paired files
        for rec_file, sup_file in zip(args.recording_files, args.supervision_files):
            if not Path(rec_file).exists():
                parser.error(f"Recording file does not exist: {rec_file}")
            if not Path(sup_file).exists():
                parser.error(f"Supervision file does not exist: {sup_file}")
            if not rec_file.endswith(('.jsonl.gz', '.jsonl')):
                parser.error(f"Recording file must be .jsonl.gz or .jsonl: {rec_file}")
            if not sup_file.endswith(('.jsonl.gz', '.jsonl')):
                parser.error(f"Supervision file must be .jsonl.gz or .jsonl: {sup_file}")
        
        input_files_count = len(args.recording_files)
    else:
        if not args.input_files:
            parser.error("Either input_files or --paired-manifests mode is required")
        
        # Validate single files
        for file_path in args.input_files:
            if not Path(file_path).exists():
                parser.error(f"Input file does not exist: {file_path}")
            if not file_path.endswith(('.jsonl.gz', '.jsonl')):
                parser.error(f"Input file must be .jsonl.gz or .jsonl: {file_path}")
        
        input_files_count = len(args.input_files)
    

    
    if args.verbose:
        if args.paired_manifests:
            print(f"Loading {len(args.recording_files)} paired CutSet files...")
        else:
            print(f"Loading {len(args.input_files)} CutSet files...")
        print(f"Output directory: {args.output_dir}")
        print(f"Output file: {args.output_name}")
    
    try:
        # Load CutSets from files
        if args.paired_manifests:
            cutsets = load_paired_cutsets_from_manifests(args.recording_files, args.supervision_files)
        else:
            cutsets = load_cutsets_from_files(args.input_files)
        
        if args.verbose:
            total_cuts = sum(len(cs) for cs in cutsets)
            print(f"Total cuts loaded: {total_cuts}")
        
        # Combine datasets
        combined_cutset = combine_data(
            cuts=cutsets,
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
