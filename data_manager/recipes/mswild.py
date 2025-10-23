"""
MSDWILD: Multi-modal Speaker Diarization Dataset in the Wild

This dataset is designed for multi-modal speaker diarization and lip-speech synchronization in the wild.
Dataset URL: https://github.com/X-LANCE/MSDWILD

The dataset includes:
- Audio files (WAVs) - 7.56 GB
- RTTM files for speaker diarization annotations
- Video files (optional) - 43.14 GB
- Cropped faces (optional) - 14.49 GB
"""

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional

import gdown
from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download
from tqdm.auto import tqdm

# Dataset URLs from the GitHub repository
AUDIO_ZIP_URL = (
    "https://drive.google.com/uc?export=download&id=1I5qfuPPGBM9keJKz0VN-OYEeRMJ7dgpl"
)
RTTM_ZIP_URL = "https://github.com/X-LANCE/MSDWILD/archive/master.zip"


def download_mswild(
    target_dir: Pathlike,
    force_download: bool = False,
    download_audio: bool = True,
    download_video: bool = False,
    download_faces: bool = False,
):
    """
    Download the MSDWILD dataset.

    Args:
        target_dir: Base directory where datasets are stored (will create mswild subdirectory)
        force_download: If True, re-download even if files exist
        download_audio: If True, download audio files (required for basic functionality)
        download_video: If True, download video files (optional, large ~43GB)
        download_faces: If True, download cropped faces (optional, ~14GB)
    """
    # Create dataset-specific directory
    dataset_dir = Path(target_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    completed_detector = dataset_dir / ".completed"

    if not completed_detector.is_file() or force_download:
        logging.info("Downloading MSDWILD dataset...")

        # Download RTTM annotations (always needed)
        logging.info("Downloading RTTM annotations...")
        resumable_download(RTTM_ZIP_URL, dataset_dir / "annotations.zip")
        with zipfile.ZipFile(dataset_dir / "annotations.zip") as zip_f:
            zip_f.extractall(dataset_dir)

        # Move RTTM files to the main directory
        annotations_dir = dataset_dir / "MSDWILD-master"
        if annotations_dir.exists():
            # Copy RTTM files
            rttm_dir = dataset_dir / "rttms"
            rttm_dir.mkdir(exist_ok=True)
            if (annotations_dir / "rttms").exists():
                shutil.copytree(annotations_dir / "rttms", rttm_dir, dirs_exist_ok=True)

            # Clean up
            shutil.rmtree(annotations_dir)

            # Download audio files if requested
            if download_audio:
                logging.info("Downloading audio files...")
                audio_zip_path = dataset_dir / "wavs.zip"

                # Check if audio zip already exists and is complete
                if audio_zip_path.exists() and audio_zip_path.stat().st_size > 0:
                    logging.info(f"Audio zip already exists: {audio_zip_path}")
                else:
                    # Use gdown to download from Google Drive with retry logic
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            logging.info(
                                f"Download attempt {attempt + 1}/{max_retries}"
                            )
                            gdown.download(
                                "https://drive.google.com/uc?id=1I5qfuPPGBM9keJKz0VN-OYEeRMJ7dgpl",
                                str(audio_zip_path),
                                quiet=False,
                            )
                            break
                        except Exception as e:
                            logging.warning(
                                f"Download attempt {attempt + 1} failed: {e}"
                            )
                            if attempt < max_retries - 1:
                                logging.info("Retrying download...")
                                if audio_zip_path.exists():
                                    audio_zip_path.unlink()  # Remove partial file
                            else:
                                logging.error(
                                    "All download attempts failed. Please download manually."
                                )
                                logging.error(
                                    "Manual download URL: https://drive.google.com/file/d/1I5qfuPPGBM9keJKz0VN-OYEeRMJ7dgpl"
                                )
                                raise

                # Extract audio files
                wavs_dir = dataset_dir / "wavs"
                wavs_dir.mkdir(exist_ok=True)

                # Check if extraction is needed
                if not any(wavs_dir.iterdir()):
                    logging.info("Extracting audio files...")
                    with zipfile.ZipFile(audio_zip_path) as zip_f:
                        zip_f.extractall(wavs_dir)
                else:
                    logging.info("Audio files already extracted")

                # Clean up zip file
                audio_zip_path.unlink()

        # Download video files if requested
        if download_video:
            logging.info("Downloading video files...")
            logging.warning(
                "Video download URL needs to be obtained from the dataset authors."
            )
            logging.warning(
                "Please download the video files manually from the Google Drive link provided in the repository."
            )

        # Download cropped faces if requested
        if download_faces:
            logging.info("Downloading cropped faces...")
            logging.warning(
                "Cropped faces download URL needs to be obtained from the dataset authors."
            )
            logging.warning(
                "Please download the cropped faces manually from the Google Drive link provided in the repository."
            )

        # Clean up
        (dataset_dir / "annotations.zip").unlink(missing_ok=True)
        completed_detector.touch()

    logging.info("MSDWILD download completed.")
    return dataset_dir


def prepare_mswild(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    splits: Optional[Dict[str, str]] = None,
    sampling_rate: Optional[int] = None,
):
    """
    Prepare the MSDWILD dataset for use with lhotse.

    Args:
        corpus_dir: Directory containing the downloaded MSDWILD dataset
        output_dir: Directory where the prepared manifests will be saved
        splits: Dictionary mapping split names to RTTM file patterns
                Default: {"train": "few_train", "dev": "few_val", "test": "many_val"}

    Returns:
        Dictionary with RecordingSet and SupervisionSet for each split
    """
    corpus_dir = Path(corpus_dir)

    if splits is None:
        splits = {"train": "few_train", "dev": "few_val", "test": "many_val"}

    manifests = {}

    for split_name, rttm_pattern in splits.items():
        logging.info(f"Preparing {split_name} split...")

        # Find RTTM files for this split
        rttm_dir = corpus_dir / "rttms"
        if not rttm_dir.exists():
            raise ValueError(f"RTTM directory not found: {rttm_dir}")

        # Look for RTTM files matching the pattern
        rttm_files = list(rttm_dir.glob(f"*{rttm_pattern}*.rttm"))
        if not rttm_files:
            # Fallback: look for any RTTM files
            rttm_files = list(rttm_dir.glob("*.rttm"))
            if not rttm_files:
                logging.warning(f"No RTTM files found for {split_name} split")
                continue

        recordings = []
        supervisions = []
        processed_audio_ids = set()  # Track processed audio files to avoid duplicates

        for rttm_file in tqdm(rttm_files, desc=f"Processing {split_name}"):
            # Read RTTM file to get audio IDs and segments
            audio_segments = {}  # audio_id -> list of segments

            with open(rttm_file, "r") as f:
                for line in f:
                    if line.startswith("SPEAKER"):
                        parts = line.split()
                        if len(parts) >= 8:
                            audio_id = parts[1]
                            start = float(parts[3])
                            duration = float(parts[4])
                            speaker = parts[7]

                            if audio_id not in audio_segments:
                                audio_segments[audio_id] = []
                            audio_segments[audio_id].append((start, duration, speaker))

            # Process each audio file referenced in the RTTM
            for audio_id, segments in audio_segments.items():
                # Skip if we've already processed this audio file
                if audio_id in processed_audio_ids:
                    continue

                audio_file = corpus_dir / "wavs" / "wav" / f"{audio_id}.wav"

                # Check if audio file exists
                if not audio_file.exists():
                    logging.warning(f"Audio file not found: {audio_file}")
                    continue

                # Create recording (resample first if needed)
                if sampling_rate:
                    from data_manager.recipes.audio_utils import resample_if_needed

                    audio_to_use = resample_if_needed(audio_file, int(sampling_rate))
                    recording = Recording.from_file(str(audio_to_use))
                else:
                    recording = Recording.from_file(audio_file)
                recordings.append(recording)
                processed_audio_ids.add(audio_id)

                # Create supervisions for this audio file
                for segment_idx, (start, duration, speaker) in enumerate(segments):
                    supervisions.append(
                        SupervisionSegment(
                            id=f"{audio_id}-{segment_idx}",
                            recording_id=audio_id,
                            start=start,
                            duration=duration,
                            channel=0,
                            speaker=speaker,
                        )
                    )

        if not recordings:
            logging.warning(f"No recordings found for {split_name} split")
            continue

        # Create sets
        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)

        # Fix and validate manifests
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        # Save manifests if output_dir is specified
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            recording_set.to_file(
                output_dir / f"msdwild_recordings_{split_name}.jsonl.gz"
            )
            supervision_set.to_file(
                output_dir / f"msdwild_supervisions_{split_name}.jsonl.gz"
            )

        manifests[split_name] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    return manifests


def _read_rttm(filename: Pathlike):
    """
    Read RTTM file and yield (start, duration, speaker) tuples.

    RTTM format: SPEAKER file 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
    """
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("SPEAKER"):
                parts = line.split()
                if len(parts) >= 8:
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]
                    yield start, duration, speaker
