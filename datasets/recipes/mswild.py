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
        target_dir: Directory where the dataset will be downloaded
        force_download: If True, re-download even if files exist
        download_audio: If True, download audio files (required for basic functionality)
        download_video: If True, download video files (optional, large ~43GB)
        download_faces: If True, download cropped faces (optional, ~14GB)
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    completed_detector = target_dir / ".completed"

    if not completed_detector.is_file() or force_download:
        logging.info("Downloading MSDWILD dataset...")

        # Download RTTM annotations (always needed)
        logging.info("Downloading RTTM annotations...")
        resumable_download(RTTM_ZIP_URL, target_dir / "annotations.zip")
        with zipfile.ZipFile(target_dir / "annotations.zip") as zip_f:
            zip_f.extractall(target_dir)

        # Move RTTM files to the main directory
        annotations_dir = target_dir / "MSDWILD-master"
        if annotations_dir.exists():
            # Copy RTTM files
            rttm_dir = target_dir / "rttms"
            rttm_dir.mkdir(exist_ok=True)
            if (annotations_dir / "rttms").exists():
                shutil.copytree(annotations_dir / "rttms", rttm_dir, dirs_exist_ok=True)

            # Clean up
            shutil.rmtree(annotations_dir)

        # Download audio files if requested
        if download_audio:
            logging.info("Downloading audio files...")
            audio_zip_path = target_dir / "wavs.zip"
            resumable_download(AUDIO_ZIP_URL, audio_zip_path)

            # Extract audio files
            wavs_dir = target_dir / "wavs"
            wavs_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(audio_zip_path) as zip_f:
                zip_f.extractall(wavs_dir)

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
        (target_dir / "annotations.zip").unlink(missing_ok=True)
        completed_detector.touch()

    logging.info("MSDWILD download completed.")


def prepare_mswild(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    splits: Optional[Dict[str, str]] = None,
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

        for rttm_file in tqdm(rttm_files, desc=f"Processing {split_name}"):
            # Extract audio file name from RTTM file
            audio_file = corpus_dir / "wavs" / f"{rttm_file.stem}.wav"

            # Check if audio file exists
            if not audio_file.exists():
                logging.warning(f"Audio file not found: {audio_file}")
                continue

            # Create recording
            recording = Recording.from_file(audio_file)
            recordings.append(recording)

            # Read RTTM file and create supervisions
            for segment_idx, (start, duration, speaker) in enumerate(
                _read_rttm(rttm_file)
            ):
                supervisions.append(
                    SupervisionSegment(
                        id=f"{rttm_file.stem}-{segment_idx}",
                        recording_id=rttm_file.stem,
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
