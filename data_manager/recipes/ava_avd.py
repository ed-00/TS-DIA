import logging
import subprocess
import tarfile
from pathlib import Path
from typing import Dict, Optional

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike
from tqdm.auto import tqdm

# Dataset URLs and IDs from the AVA-AVD repository
ANNOTATIONS_GDRIVE_ID = "18kjJJbebBg7e8umI6HoGE4_tI3OWufzA"
VIDEOS_BASE_URL = "https://s3.amazonaws.com/ava-dataset/trainval/"
REPOSITORY_URL = "https://github.com/zcxu-eric/AVA-AVD/archive/main.zip"


def download_ava_avd(
    target_dir: Pathlike,
    force_download: bool = False,
    download_annotations: bool = True,
    download_videos: bool = True,
) -> Pathlike:
    """
    Download the AVA-AVD dataset.

    Args:
        target_dir: Base directory where datasets are stored (will create ava_avd subdirectory)
        force_download: If True, re-download even if files exist
        download_annotations: If True, download annotation files using gdown
        download_videos: If True, download video files using wget
    """
    # Create dataset-specific directory
    dataset_dir = Path(target_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    completed_detector = dataset_dir / ".completed"

    if not completed_detector.is_file() or force_download:
        logging.info("Downloading AVA-AVD dataset...")

        # Download repository to get video list and scripts
        logging.info("Downloading AVA-AVD repository...")
        repo_zip_path = dataset_dir / "ava_avd_repo.zip"

        # Use wget to download the repository
        cmd = f"wget -O {repo_zip_path} {REPOSITORY_URL}"
        subprocess.call(cmd, shell=True)

        # Extract repository
        import zipfile

        with zipfile.ZipFile(repo_zip_path) as zip_f:
            zip_f.extractall(dataset_dir)

        # Move dataset files to the main directory
        repo_dir = dataset_dir / "AVA-AVD-main"
        if repo_dir.exists():
            # Copy dataset directory
            if (repo_dir / "dataset").exists():
                import shutil

                shutil.copytree(
                    repo_dir / "dataset", dataset_dir / "dataset", dirs_exist_ok=True
                )

            # Clean up
            shutil.rmtree(repo_dir)

        # Create videos directory
        videos_dir = dataset_dir / "dataset" / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        # Download videos if requested
        if download_videos:
            logging.info("Downloading videos...")
            video_list_file = dataset_dir / "dataset" / "split" / "video.list"
            if video_list_file.exists():
                with open(video_list_file, "r") as f:
                    videos = f.readlines()

                for i, video in enumerate(videos):
                    video_name = video.strip()
                    video_path = videos_dir / video_name

                    if video_path.exists() and not force_download:
                        logging.info(f"Video {video_name} already exists, skipping...")
                        continue

                    logging.info(f"Downloading {video_name} [{i + 1}/{len(videos)}]")
                    cmd = f"wget -P {videos_dir} {VIDEOS_BASE_URL}{video_name}"
                    subprocess.call(cmd, shell=True)
            else:
                logging.warning(f"Video list file not found: {video_list_file}")

        # Download annotations if requested
        if download_annotations:
            logging.info("Downloading annotations...")
            annotations_dir = dataset_dir / "dataset" / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)

            # Download annotations using gdown
            cmd = f"gdown --id {ANNOTATIONS_GDRIVE_ID} -O {dataset_dir}/annotations.tar.gz"
            subprocess.call(cmd, shell=True)

            # Extract annotations
            if (dataset_dir / "annotations.tar.gz").exists():
                with tarfile.open(dataset_dir / "annotations.tar.gz", "r") as tar:
                    tar.extractall(dataset_dir / "dataset")

                # Clean up tar file
                (dataset_dir / "annotations.tar.gz").unlink()

        # Clean up repository zip
        repo_zip_path.unlink(missing_ok=True)
        completed_detector.touch()

    logging.info("AVA-AVD download completed.")
    return dataset_dir


def prepare_ava_avd(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    splits: Optional[Dict[str, str]] = None,
    sampling_rate: int = 16000,  # Allow configurable sampling rate
) -> Dict[str, Dict[str, RecordingSet | SupervisionSet]]:
    """
    Prepare the AVA-AVD dataset for use with lhotse.

    Args:
        corpus_dir: Directory containing the downloaded AVA-AVD dataset
        output_dir: Directory where the prepared manifests will be saved
        splits: Dictionary mapping split names to subdirectories
                Default: {"train": "train", "val": "val", "test": "test"}

    Returns:
        Dictionary with RecordingSet and SupervisionSet for each split
    """
    corpus_dir = Path(corpus_dir)

    if splits is None:
        splits = {"train": "train", "val": "val", "test": "test"}

    manifests = {}

    for split_name, split_dir in splits.items():
        logging.info(f"Preparing {split_name} split...")

        # Find annotation files for this split (try both labs and rttms)
        labs_dir = corpus_dir / "dataset" / "labs"
        rttms_dir = corpus_dir / "dataset" / "rttms"

        if not labs_dir.exists() and not rttms_dir.exists():
            logging.warning(
                f"Annotation directories not found: {labs_dir} or {rttms_dir}"
            )
            continue

        # Find video files for this split
        videos_dir = corpus_dir / "dataset" / "videos"
        if not videos_dir.exists():
            logging.warning(f"Videos directory not found: {videos_dir}")
            continue

        recordings = []
        supervisions = []
        processed_video_ids = set()  # Track processed video files to avoid duplicates

        # Get split list to filter videos
        split_list_file = corpus_dir / "dataset" / "split" / f"{split_dir}.list"
        if not split_list_file.exists():
            logging.warning(f"Split list file not found: {split_list_file}")
            continue

        # Read video list for this split
        with open(split_list_file, "r") as f:
            split_video_entries = [line.strip() for line in f.readlines()]

        # Get all annotation files (try lab files first, then RTTM files)
        annotation_files = []
        if labs_dir.exists():
            annotation_files = list(labs_dir.glob("*.lab"))
        elif rttms_dir.exists():
            annotation_files = list(rttms_dir.glob("*.rttm"))

        if not annotation_files:
            logging.warning(f"No annotation files found for {split_name} split")
            continue

        for annotation_file in tqdm(annotation_files, desc=f"Processing {split_name}"):
            # Extract video ID from annotation filename (e.g., "vfjywN5CN0Y_c_02.lab" -> "vfjywN5CN0Y")
            video_id = annotation_file.stem.split("_")[0]  # Remove suffix like "_c_02"

            # Check if this video ID is in the split (need to check against base video IDs)
            # Split entries are like "video_id_c_01", "video_id_c_02", etc.
            video_in_split = any(
                entry.startswith(video_id + "_c_") for entry in split_video_entries
            )
            if not video_in_split:
                continue

            # Skip if we've already processed this video file
            if video_id in processed_video_ids:
                continue

            # Find corresponding video file (try common video extensions)
            video_file = None
            for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                potential_video = videos_dir / f"{video_id}{ext}"
                if potential_video.exists():
                    video_file = potential_video
                    break

            if video_file is None:
                logging.warning(f"Video file not found for: {video_id}")
                continue

            # Create recording from video file with faster method
            try:
                # Use ffprobe directly for faster metadata extraction
                import json
                import subprocess

                # Get video info using ffprobe (much faster than TorchAudio)
                cmd = [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    str(video_file),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    info = json.loads(result.stdout)
                    duration = float(info["format"]["duration"])
                    # Use the sampling_rate parameter instead of hardcoded value
                    sample_rate = sampling_rate
                    num_channels = 1  # Default mono

                    # Extract audio from video to WAV file
                    # This fixes the Lhotse AudioSource type='video' issue
                    audio_dir = corpus_dir / "audio"
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    audio_file = audio_dir / f"{video_id}.wav"

                    # Extract audio if not already extracted
                    if not audio_file.exists():
                        logging.info(
                            f"Extracting audio from {video_id} at {sample_rate}Hz..."
                        )
                        extract_cmd = [
                            "ffmpeg",
                            "-i",
                            str(video_file),
                            "-vn",  # No video
                            "-acodec",
                            "pcm_s16le",  # 16-bit PCM
                            "-ar",
                            str(sample_rate),  # Resample to configured rate
                            "-ac",
                            str(num_channels),  # Mono
                            "-y",  # Overwrite if exists
                            str(audio_file),
                        ]
                        try:
                            subprocess.run(
                                extract_cmd,
                                check=True,
                                capture_output=True,
                                timeout=300,  # 5 minute timeout per file
                            )
                            logging.info(f"Audio extracted: {audio_file}")
                        except subprocess.CalledProcessError as e:
                            logging.warning(
                                f"Failed to extract audio for {video_id}: {e.stderr.decode()}"
                            )
                            continue
                        except subprocess.TimeoutExpired:
                            logging.warning(
                                f"Audio extraction timed out for {video_id}"
                            )
                            continue

                    # Create recording with extracted audio file
                    from lhotse.audio import AudioSource

                    audio_source = AudioSource(
                        type="file",  # Use 'file' instead of 'video'
                        channels=list(range(num_channels)),
                        source=str(audio_file),
                    )

                    recording = Recording(
                        id=video_id,
                        sources=[audio_source],
                        sampling_rate=sample_rate,
                        num_samples=int(duration * sample_rate),
                        duration=duration,
                    )
                    recordings.append(recording)
                    processed_video_ids.add(video_id)
                    logging.info(
                        f"Created recording: {video_id}, duration: {duration:.2f}s"
                    )
                else:
                    logging.warning(f"ffprobe failed for {video_file}: {result.stderr}")
                    continue

            except Exception as e:
                logging.warning(f"Error creating recording for {video_id}: {e}")
                continue

            # Read annotation file to get segments
            try:
                with open(annotation_file, "r") as f:
                    for line_idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue

                        # Handle lab format
                        if annotation_file.suffix.lower() == ".lab":
                            parts = line.split()
                            if len(parts) >= 3:
                                start_time = float(parts[0])
                                end_time = float(parts[1])
                                label = parts[2]

                                # Only process speech segments
                                if label == "speech":
                                    duration = end_time - start_time
                                    supervisions.append(
                                        SupervisionSegment(
                                            id=f"{video_id}-{line_idx}",
                                            recording_id=video_id,
                                            start=start_time,
                                            duration=duration,
                                            channel=0,
                                            speaker="speech",  # AVA-AVD doesn't have speaker labels
                                        )
                                    )

                        # Handle RTTM format
                        elif annotation_file.suffix.lower() == ".rttm":
                            if line.startswith("SPEAKER"):
                                parts = line.split()
                                if len(parts) >= 8:
                                    start_time = float(parts[3])
                                    duration = float(parts[4])
                                    speaker = parts[7]

                                    supervisions.append(
                                        SupervisionSegment(
                                            id=f"{video_id}-{line_idx}",
                                            recording_id=video_id,
                                            start=start_time,
                                            duration=duration,
                                            channel=0,
                                            speaker=speaker,
                                        )
                                    )

            except Exception as e:
                logging.warning(
                    f"Error processing annotation file {annotation_file}: {e}"
                )
                continue

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
                output_dir / f"ava_avd_recordings_{split_name}.jsonl.gz"
            )
            supervision_set.to_file(
                output_dir / f"ava_avd_supervisions_{split_name}.jsonl.gz"
            )

        manifests[split_name] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    return manifests
