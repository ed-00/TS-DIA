import json
import logging
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile
import shutil
import soundfile as sf

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

        with zipfile.ZipFile(repo_zip_path) as zip_f:
            zip_f.extractall(dataset_dir)

        # Move dataset files to the main directory
        repo_dir = dataset_dir / "AVA-AVD-main"
        if repo_dir.exists():
            # Copy dataset directory
            if (repo_dir / "dataset").exists():
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
                        logging.info(
                            f"Video {video_name} already exists, skipping...")
                        continue

                    logging.info(
                        f"Downloading {video_name} [{i + 1}/{len(videos)}]")
                    cmd = f"wget -P {videos_dir} {VIDEOS_BASE_URL}{video_name}"
                    subprocess.call(cmd, shell=True)
            else:
                logging.warning(
                    f"Video list file not found: {video_list_file}")

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

    manifests: Dict[str, Dict[str, RecordingSet | SupervisionSet]] = {}

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

        recordings: List[Recording] = []
        supervisions: List[SupervisionSegment] = []
        recordings_map: Dict[str, Recording] = {}
        failed_videos: set[str] = set()
        failed_chunks: set[str] = set()
        video_path_cache: Dict[str, Path] = {}
        video_duration_cache: Dict[str, float] = {}

        audio_root = corpus_dir / "audio" / split_name
        audio_root.mkdir(parents=True, exist_ok=True)

        chunk_pattern = re.compile(r"_c_(\d+)$")
        base_chunk_offset = 900.0
        chunk_stride = 300.0
        start_pad = 1.0
        end_pad = 1.0

        # Get split list to filter videos
        split_list_file = corpus_dir / "dataset" / \
            "split" / f"{split_dir}.list"
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
            logging.warning(
                f"No annotation files found for {split_name} split")
            continue

        for annotation_file in tqdm(annotation_files, desc=f"Processing {split_name}"):
            chunk_id = annotation_file.stem

            if chunk_id not in split_video_entries:
                continue

            base_video_id = re.sub(r"_c_\d+$", "", chunk_id)
            if not base_video_id:
                logging.warning(
                    f"Unable to infer base video id from annotation {annotation_file.name}"
                )
                continue

            if base_video_id in failed_videos or chunk_id in failed_chunks:
                continue

            chunk_match = chunk_pattern.search(chunk_id)
            if not chunk_match:
                logging.warning(
                    "Could not parse chunk index from %s", annotation_file.name
                )
                failed_chunks.add(chunk_id)
                continue

            try:
                segments: List[Tuple[int, float, float, str]] = []
                min_start = float("inf")
                max_end = 0.0
                with open(annotation_file, "r") as f:
                    for line_idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue

                        if annotation_file.suffix.lower() == ".lab":
                            parts = line.split()
                            if len(parts) < 3:
                                continue
                            start_time = float(parts[0])
                            end_time = float(parts[1])
                            label = parts[2]
                            if label != "speech":
                                continue
                            segments.append(
                                (line_idx, start_time, end_time, "speech"))
                        elif annotation_file.suffix.lower() == ".rttm":
                            if not line.startswith("SPEAKER"):
                                continue
                            parts = line.split()
                            if len(parts) < 8:
                                continue
                            start_time = float(parts[3])
                            duration = float(parts[4])
                            end_time = start_time + duration
                            speaker = parts[7]
                            segments.append(
                                (line_idx, start_time, end_time, speaker))
                        else:
                            continue

                        min_start = min(min_start, start_time)
                        max_end = max(max_end, end_time)
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "Error reading annotation file %s: %s", annotation_file, exc
                )
                failed_chunks.add(chunk_id)
                continue

            if not segments:
                continue

            chunk_index = int(chunk_match.group(1))
            expected_offset = base_chunk_offset + \
                (chunk_index - 1) * chunk_stride
            chunk_start = min(expected_offset, min_start) - start_pad
            if chunk_start < 0.0:
                chunk_start = 0.0
            chunk_end_expected = expected_offset + chunk_stride
            chunk_end = max(chunk_end_expected, max_end) + end_pad

            # Locate video file and metadata
            video_file = video_path_cache.get(base_video_id)
            if video_file is None:
                for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                    candidate = videos_dir / f"{base_video_id}{ext}"
                    if candidate.exists():
                        video_file = candidate
                        video_path_cache[base_video_id] = candidate
                        break
                if video_file is None:
                    logging.warning(
                        f"Video file not found for: {base_video_id}")
                    failed_videos.add(base_video_id)
                    failed_chunks.add(chunk_id)
                    continue

            video_duration = video_duration_cache.get(base_video_id)
            if video_duration is None:
                cmd = [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    str(video_file),
                ]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    logging.warning(
                        "ffprobe failed for %s: %s", video_file, result.stderr
                    )
                    failed_videos.add(base_video_id)
                    failed_chunks.add(chunk_id)
                    continue
                info = json.loads(result.stdout)
                video_duration = float(info["format"]["duration"])
                video_duration_cache[base_video_id] = video_duration

            if chunk_start >= video_duration:
                logging.warning(
                    "Chunk %s starts beyond video duration (%.2f ≥ %.2f)",
                    chunk_id,
                    chunk_start,
                    video_duration,
                )
                failed_chunks.add(chunk_id)
                continue

            chunk_end = min(chunk_end, video_duration)
            if chunk_end <= chunk_start:
                logging.warning(
                    "Chunk %s has non-positive duration after clamping (start=%.2f, end=%.2f)",
                    chunk_id,
                    chunk_start,
                    chunk_end,
                )
                failed_chunks.add(chunk_id)
                continue

            chunk_duration = chunk_end - chunk_start
            audio_file = audio_root / f"{chunk_id}.wav"

            need_extract = True
            if audio_file.exists():
                try:
                    with sf.SoundFile(audio_file) as snd_file:
                        existing_sr = snd_file.samplerate
                        existing_duration = snd_file.frames / snd_file.samplerate
                    duration_diff = abs(existing_duration - chunk_duration)
                    if existing_sr == sampling_rate and duration_diff <= 0.25:
                        need_extract = False
                except Exception:  # noqa: BLE001
                    need_extract = True

            if need_extract:
                if audio_file.exists():
                    try:
                        audio_file.unlink()
                    except OSError as exc:  # noqa: BLE001
                        logging.warning(
                            "Unable to remove stale audio file %s: %s",
                            audio_file,
                            exc,
                        )
                        failed_chunks.add(chunk_id)
                        continue

                logging.info(
                    "Extracting audio for %s (%.2fs @ %d Hz)",
                    chunk_id,
                    chunk_duration,
                    sampling_rate,
                )
                extract_cmd = [
                    "ffmpeg",
                    "-y",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-i",
                    str(video_file),
                    "-ss",
                    f"{chunk_start:.3f}",
                    "-t",
                    f"{chunk_duration:.3f}",
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(sampling_rate),
                    "-ac",
                    "1",
                    str(audio_file),
                ]
                try:
                    subprocess.run(
                        extract_cmd,
                        check=True,
                        capture_output=True,
                        timeout=300,
                    )
                except subprocess.CalledProcessError as exc:  # noqa: PERF203
                    logging.warning(
                        "Failed to extract chunk %s: %s",
                        chunk_id,
                        exc.stderr.decode() if exc.stderr else exc,
                    )
                    failed_chunks.add(chunk_id)
                    continue
                except subprocess.TimeoutExpired:
                    logging.warning(
                        "Audio extraction timed out for %s", chunk_id)
                    failed_chunks.add(chunk_id)
                    continue

            if chunk_id not in recordings_map:
                try:
                    recording = Recording.from_file(
                        audio_file, recording_id=chunk_id)
                except Exception as exc:  # noqa: BLE001
                    logging.warning(
                        "Failed to create recording for %s: %s", chunk_id, exc)
                    failed_chunks.add(chunk_id)
                    continue
                recordings.append(recording)
                recordings_map[chunk_id] = recording

            recording = recordings_map[chunk_id]
            recording_duration = recording.duration

            for line_idx, start_time, end_time, speaker in segments:
                start_in_chunk = start_time - chunk_start
                end_in_chunk = end_time - chunk_start
                if end_in_chunk <= start_in_chunk:
                    continue
                start_in_chunk = max(0.0, start_in_chunk)
                end_in_chunk = min(recording_duration, max(0.0, end_in_chunk))
                duration = end_in_chunk - start_in_chunk
                if duration <= 0:
                    continue
                supervisions.append(
                    SupervisionSegment(
                        id=f"{chunk_id}-{line_idx}",
                        recording_id=chunk_id,
                        start=start_in_chunk,
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
        recording_set, supervision_set = fix_manifests(
            recording_set, supervision_set)
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
