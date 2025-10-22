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

# Dataset URLs from the VoxConverse repository
DEV_AUDIO_URL = (
    "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip"
)
TEST_AUDIO_URL = (
    "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip"
)
ANNOTATIONS_URL = "https://github.com/joonson/voxconverse/archive/master.zip"


def download_voxconverse(
    target_dir: Pathlike,
    force_download: bool = False,
    download_dev: bool = True,
    download_test: bool = True,
):
    """
    Download the VoxConverse dataset.

    Args:
        target_dir: Base directory where datasets are stored (will create voxconverse subdirectory)
        force_download: If True, re-download even if files exist
        download_dev: If True, download development set audio files
        download_test: If True, download test set audio files
    """
    # Create dataset-specific directory
    dataset_dir = Path(target_dir) / "voxconverse"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    completed_detector = dataset_dir / ".completed"

    if not completed_detector.is_file() or force_download:
        logging.info("Downloading VoxConverse dataset...")

        # Download annotations (always needed)
        logging.info("Downloading RTTM annotations...")
        annotations_zip_path = dataset_dir / "annotations.zip"
        resumable_download(ANNOTATIONS_URL, annotations_zip_path)

        with zipfile.ZipFile(annotations_zip_path) as zip_f:
            zip_f.extractall(dataset_dir)

        # Move RTTM files to the main directory
        annotations_dir = dataset_dir / "voxconverse-master"
        if annotations_dir.exists():
            # Copy RTTM files from dev and test folders
            rttm_dir = dataset_dir / "rttms"
            rttm_dir.mkdir(exist_ok=True)

            # Copy dev RTTM files
            dev_rttm_dir = rttm_dir / "dev"
            dev_rttm_dir.mkdir(exist_ok=True)
            if (annotations_dir / "dev").exists():
                for rttm_file in (annotations_dir / "dev").glob("*.rttm"):
                    shutil.copy2(rttm_file, dev_rttm_dir)

            # Copy test RTTM files
            test_rttm_dir = rttm_dir / "test"
            test_rttm_dir.mkdir(exist_ok=True)
            if (annotations_dir / "test").exists():
                for rttm_file in (annotations_dir / "test").glob("*.rttm"):
                    shutil.copy2(rttm_file, test_rttm_dir)

            # Clean up
            shutil.rmtree(annotations_dir)

        # Download dev audio files if requested
        if download_dev:
            logging.info("Downloading development set audio files...")
            dev_audio_zip_path = dataset_dir / "dev_wav.zip"
            resumable_download(DEV_AUDIO_URL, dev_audio_zip_path)

            # Extract dev audio files
            dev_wavs_dir = dataset_dir / "wavs" / "dev"
            dev_wavs_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(dev_audio_zip_path) as zip_f:
                zip_f.extractall(dev_wavs_dir)

            # Clean up zip file
            dev_audio_zip_path.unlink()

        # Download test audio files if requested
        if download_test:
            logging.info("Downloading test set audio files...")
            test_audio_zip_path = dataset_dir / "test_wav.zip"
            resumable_download(TEST_AUDIO_URL, test_audio_zip_path)

            # Extract test audio files
            test_wavs_dir = dataset_dir / "wavs" / "test"
            test_wavs_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(test_audio_zip_path) as zip_f:
                zip_f.extractall(test_wavs_dir)

            # Clean up zip file
            test_audio_zip_path.unlink()

        # Clean up annotations zip
        annotations_zip_path.unlink(missing_ok=True)
        completed_detector.touch()

    logging.info("VoxConverse download completed.")
    return dataset_dir


def prepare_voxconverse(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    splits: Optional[Dict[str, str]] = None,
    sampling_rate: Optional[int] = None,
):
    """
    Prepare the VoxConverse dataset for use with lhotse.

    Args:
        corpus_dir: Directory containing the downloaded VoxConverse dataset
        output_dir: Directory where the prepared manifests will be saved
        splits: Dictionary mapping split names to subdirectories
                Default: {"dev": "dev", "test": "test"}
        sampling_rate: If specified, resample audio to this sampling rate

    Returns:
        Dictionary with RecordingSet and SupervisionSet for each split
    """
    corpus_dir = Path(corpus_dir)

    if splits is None:
        splits = {"dev": "dev", "test": "test"}

    manifests = {}

    for split_name, split_dir in splits.items():
        logging.info(f"Preparing {split_name} split...")

        # Find RTTM files for this split
        rttm_dir = corpus_dir / "rttms" / split_dir
        if not rttm_dir.exists():
            logging.warning(f"RTTM directory not found: {rttm_dir}")
            continue

        # Find audio files for this split
        audio_dir = corpus_dir / "wavs" / split_dir
        if not audio_dir.exists():
            logging.warning(f"Audio directory not found: {audio_dir}")
            continue

        recordings = []
        supervisions = []
        processed_audio_ids = set()  # Track processed audio files to avoid duplicates

        # Get all RTTM files
        rttm_files = list(rttm_dir.glob("*.rttm"))
        if not rttm_files:
            logging.warning(f"No RTTM files found for {split_name} split")
            continue

        for rttm_file in tqdm(rttm_files, desc=f"Processing {split_name}"):
            # Extract audio ID from RTTM filename (e.g., "abc123.rttm" -> "abc123")
            audio_id = rttm_file.stem

            # Skip if we've already processed this audio file
            if audio_id in processed_audio_ids:
                continue

            # Find corresponding audio file (check both direct and audio subdirectory)

            audio_file = audio_dir / "audio" / f"{audio_id}.wav"
            if not audio_file.exists():
                logging.warning(f"Audio file not found: {audio_file}")
                continue
            # Optionally resample audio to target sampling_rate using shared helper
            from data_manager.recipes.audio_utils import resample_if_needed

            if sampling_rate:
                audio_to_use = resample_if_needed(Path(audio_file), int(sampling_rate))
            else:
                audio_to_use = Path(audio_file)

            # Create recording
            recording = Recording.from_file(str(audio_to_use))
            recordings.append(recording)
            processed_audio_ids.add(audio_id)

            # Read RTTM file to get segments
            with open(rttm_file, "r") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if line.startswith("SPEAKER"):
                        parts = line.split()
                        if len(parts) >= 8:
                            start = float(parts[3])
                            duration = float(parts[4])
                            speaker = parts[7]

                            supervisions.append(
                                SupervisionSegment(
                                    id=f"{audio_id}-{line_idx}",
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
                output_dir / f"voxconverse_recordings_{split_name}.jsonl.gz"
            )
            supervision_set.to_file(
                output_dir / f"voxconverse_supervisions_{split_name}.jsonl.gz"
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
