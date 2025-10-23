#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
LibriheavyMix Dataset Recipe

This module provides download and processing functions for the LibriheavyMix dataset,
a 20,000-hour synthesized corpus for overlapped speech separation and diarization.

The dataset is available on HuggingFace with multiple splits:
- LibriheavyMix-small: 100 hours
- LibriheavyMix-medium: 900 hours
- LibriheavyMix-large: 9,000 hours
- LibriheavyMix-dev: Development set
- LibriheavyMix-test: Test set

Each split contains mixtures with 1-4 speakers with reverberation effects.

The dataset files are organized by speaker count and processing variant:
- lsheavymix_cuts_dev_{speaker_count}spk.jsonl.gz (base files)
- lsheavymix_cuts_dev_{speaker_count}spk_snr_aug_mono.jsonl.gz (with SNR augmentation)
- lsheavymix_cuts_dev_{speaker_count}spk_snr_aug_mono_rir.jsonl.gz (with RIR)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from lhotse import MonoCut, Recording
from lhotse.audio import AudioSource
from lhotse.utils import Pathlike, resumable_download


def download_libriheavy_mix(
    target_dir: Pathlike,
    force_download: bool = False,
    dataset_parts: Union[str, List[str]] = "small",
    speaker_counts: Union[int, List[int]] = [1, 2, 3, 4],
    cache_dir: Optional[Pathlike] = None,
):
    """
    Download the LibriheavyMix dataset from HuggingFace.

    Args:
        target_dir: Base directory where datasets are stored (will create libriheavy_mix subdirectory)
        force_download: If True, re-download even if files exist
        dataset_parts: Which dataset parts to download. Options: "small", "medium", "large", "dev", "test"
                      Can be a single string or list of strings
        speaker_counts: Number of speakers per mixture. Options: 1, 2, 3, 4 or list of these values
        cache_dir: Directory to cache HuggingFace datasets (optional)

    Returns:
        Path to the dataset directory
    """
    # Create dataset-specific directory
    dataset_dir = Path(target_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Convert single string to list for uniform processing
    if isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]
    if isinstance(speaker_counts, int):
        speaker_counts = [speaker_counts]

    # Validate dataset parts
    valid_parts = {"small", "medium", "large", "dev", "test"}
    for part in dataset_parts:
        if part not in valid_parts:
            raise ValueError(
                f"Invalid dataset part '{part}'. Valid options: {valid_parts}"
            )

    # Validate speaker counts
    valid_speaker_counts = {1, 2, 3, 4}
    for count in speaker_counts:
        if count not in valid_speaker_counts:
            raise ValueError(
                f"Invalid speaker count '{count}'. Valid options: {valid_speaker_counts}"
            )

    completed_detector = dataset_dir / ".completed"

    if not completed_detector.is_file() or force_download:
        logging.info("Downloading LibriheavyMix dataset from HuggingFace...")

        # Base URL for HuggingFace files
        base_url = "https://huggingface.co/datasets/zrjin/LibriheavyMix-{part}/resolve/main/dev-lhotse/"
        audio_base_url = "https://huggingface.co/datasets/zrjin/LibriheavyMix-{part}/resolve/main/dev_{speaker_count}spk/"

        # File variants to download (in order of preference)
        file_variants = [
            "lsheavymix_cuts_dev_{speaker_count}spk_snr_aug_mono_rir_fixed.jsonl.gz",
            "lsheavymix_cuts_dev_{speaker_count}spk_snr_aug_mono_rir.jsonl.gz",
            "lsheavymix_cuts_dev_{speaker_count}spk_snr_aug_mono.jsonl.gz",
            "lsheavymix_cuts_dev_{speaker_count}spk.jsonl.gz",
        ]

        # Audio files to download
        audio_files = [
            "audio.tar.gz",
            "src.tar.gz",
        ]

        for part in dataset_parts:
            for speaker_count in speaker_counts:
                logging.info(f"Downloading LibriheavyMix-{part}-{speaker_count}spk...")
                part_dir = dataset_dir / part / f"{speaker_count}spk"
                part_dir.mkdir(parents=True, exist_ok=True)

                # Try to download each variant until one succeeds
                downloaded = False
                for variant in file_variants:
                    filename = variant.format(speaker_count=speaker_count)
                    url = base_url.format(part=part) + filename
                    local_file = part_dir / filename

                    try:
                        logging.info(f"Attempting to download: {filename}")
                        resumable_download(url, local_file)

                        # Verify the file was downloaded and is not empty
                        if local_file.exists() and local_file.stat().st_size > 0:
                            logging.info(f"Successfully downloaded: {filename}")
                            downloaded = True
                            break
                        else:
                            logging.warning(f"Downloaded file is empty: {filename}")
                            local_file.unlink(missing_ok=True)

                    except Exception as e:
                        logging.warning(f"Failed to download {filename}: {e}")
                        local_file.unlink(missing_ok=True)
                        continue

                if not downloaded:
                    logging.error(
                        f"Failed to download any variant for LibriheavyMix-{part}-{speaker_count}spk"
                    )

                # Download audio files if manifest was downloaded successfully
                if downloaded:
                    logging.info(
                        f"Downloading audio files for LibriheavyMix-{part}-{speaker_count}spk..."
                    )
                    audio_dir = part_dir / "audio"
                    audio_dir.mkdir(exist_ok=True)

                    for audio_file in audio_files:
                        try:
                            url = (
                                audio_base_url.format(
                                    part=part, speaker_count=speaker_count
                                )
                                + audio_file
                            )
                            local_audio_file = audio_dir / audio_file

                            logging.info(f"Downloading audio file: {audio_file}")
                            resumable_download(url, local_audio_file)

                            if (
                                local_audio_file.exists()
                                and local_audio_file.stat().st_size > 0
                            ):
                                logging.info(
                                    f"Successfully downloaded audio file: {audio_file}"
                                )

                                # Extract the tar.gz file
                                import tarfile

                                logging.info(f"Extracting {audio_file}...")
                                with tarfile.open(local_audio_file, "r:gz") as tar:
                                    tar.extractall(audio_dir)
                                logging.info(f"Extracted {audio_file} successfully")

                            else:
                                logging.warning(
                                    f"Downloaded audio file is empty: {audio_file}"
                                )
                                local_audio_file.unlink(missing_ok=True)

                        except Exception as e:
                            logging.warning(
                                f"Failed to download audio file {audio_file}: {e}"
                            )
                            local_audio_file.unlink(missing_ok=True)
                            continue

        completed_detector.touch()

    logging.info("LibriheavyMix download completed.")
    return dataset_dir


def prepare_libriheavy_mix(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dataset_parts: Union[str, List[str]] = "small",
    speaker_counts: Union[int, List[int]] = [1, 2, 3, 4],
    splits: Optional[Dict[str, str]] = None,
    min_speakers: int = 1,
    max_speakers: int = 4,
    sampling_rate: Optional[int] = None,
):
    """
    Prepare the LibriheavyMix dataset for use with lhotse.

    Args:
        corpus_dir: Directory containing the downloaded LibriheavyMix dataset
        output_dir: Directory where the prepared manifests will be saved
        dataset_parts: Which dataset parts to process. Options: "small", "medium", "large", "dev", "test"
        speaker_counts: Number of speakers per mixture. Options: 1, 2, 3, 4 or list of these values
        splits: Dictionary mapping split names to subdirectories
                Default: {"train": "train", "dev": "dev", "test": "test"}
        min_speakers: Minimum number of speakers to include
        max_speakers: Maximum number of speakers to include

    Returns:
        Dictionary with RecordingSet and SupervisionSet for each split
    """
    corpus_dir = Path(corpus_dir)

    # Convert single string to list for uniform processing
    if isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]
    if isinstance(speaker_counts, int):
        speaker_counts = [speaker_counts]

    # Filter speaker counts based on min/max
    filtered_speaker_counts = [
        count for count in speaker_counts if min_speakers <= count <= max_speakers
    ]

    if splits is None:
        splits = {"train": "train", "dev": "dev", "test": "test"}

    manifests = {}

    for part in dataset_parts:
        for speaker_count in filtered_speaker_counts:
            part_dir = corpus_dir / part / f"{speaker_count}spk"
            if not part_dir.exists():
                logging.warning(f"Dataset part directory not found: {part_dir}")
                continue

            logging.info(f"Preparing LibriheavyMix-{part}-{speaker_count}spk...")

            # Find the downloaded manifest file
            manifest_files = list(part_dir.glob("lsheavymix_cuts_dev_*spk*.jsonl.gz"))
            if not manifest_files:
                logging.warning(f"No manifest files found in {part_dir}")
                continue

            # Use the first available manifest file
            manifest_file = manifest_files[0]
            logging.info(f"Processing manifest file: {manifest_file}")

            try:
                # Load the Lhotse CutSet from the manifest file
                from lhotse import CutSet

                # If audio tar extracted and sampling_rate provided, resample audio files
                if sampling_rate is not None:
                    audio_root = part_dir / "audio"
                    if audio_root.exists():
                        from data_manager.recipes.audio_utils import resample_dir

                        resample_dir(audio_root, int(sampling_rate))

                cut_set = CutSet.from_file(manifest_file)

                logging.info(f"Loaded CutSet with {len(cut_set)} cuts")

                # Update audio file paths to point to local downloaded files
                audio_dir = part_dir / "audio"
                if audio_dir.exists():
                    logging.info(
                        f"Updating audio file paths to local directory: {audio_dir}"
                    )

                    # Create a mapping from remote paths to local paths
                    updated_cuts = []
                    for cut in cut_set:
                        # Update the recording source path
                        if cut.recording.sources:
                            old_source = cut.recording.sources[0].source
                            # Extract filename from the old path
                            filename = Path(old_source).name
                            # Create new local path - the audio files are in audio/dev_2spk/ subdirectory
                            new_source = str(
                                audio_dir
                                / "audio"
                                / f"dev_{speaker_count}spk"
                                / filename
                            )

                            # Create new recording with updated source
                            new_recording = Recording(
                                id=cut.recording.id,
                                sources=[
                                    AudioSource(
                                        type="file",
                                        channels=cut.recording.sources[0].channels,
                                        source=new_source,
                                    )
                                ],
                                sampling_rate=cut.recording.sampling_rate,
                                num_samples=cut.recording.num_samples,
                                duration=cut.recording.duration,
                                channel_ids=cut.recording.channel_ids,
                            )

                            # Create new cut with updated recording
                            new_cut = MonoCut(
                                id=cut.id,
                                start=cut.start,
                                duration=cut.duration,
                                channel=cut.channel,
                                supervisions=cut.supervisions,
                                recording=new_recording,
                            )
                            updated_cuts.append(new_cut)
                        else:
                            updated_cuts.append(cut)

                    # Create new CutSet with updated paths
                    cut_set = CutSet.from_cuts(updated_cuts)
                    logging.info(
                        f"Updated {len(updated_cuts)} cuts with local audio paths"
                    )

                # Save CutSet if output_dir is specified
                if output_dir is not None:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    cut_set.to_file(
                        output_dir
                        / f"libriheavy_mix_cuts_{part}_{speaker_count}spk.jsonl.gz"
                    )

                manifests[f"{part}_{speaker_count}spk"] = {
                    "cuts": cut_set,
                }

                logging.info(f"Successfully processed CutSet with {len(cut_set)} cuts")

            except Exception as e:
                logging.error(f"Failed to process manifest file {manifest_file}: {e}")
                continue

    return manifests


def _load_libriheavy_mix_item(item_data: Dict) -> Dict:
    """
    Load and parse a single LibriheavyMix dataset item.

    Args:
        item_data: Dictionary containing the item data from HuggingFace

    Returns:
        Dictionary with parsed recording and supervision information
    """
    result = {
        "recording_id": item_data.get("id", ""),
        "audio_path": item_data.get("audio", {}).get("path", ""),
        "supervisions": [],
    }

    # Parse supervision segments
    for seg in item_data.get("supervisions", []):
        supervision = {
            "start": seg.get("start", 0.0),
            "duration": seg.get("duration", 0.0),
            "speaker": seg.get("speaker", ""),
            "text": seg.get("text", ""),
        }
        result["supervisions"].append(supervision)

    return result
