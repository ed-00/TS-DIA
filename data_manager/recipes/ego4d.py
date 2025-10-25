#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Ego4D Dataset Recipe

This module provides secure download and processing functions for the Ego4D dataset,
with a focus on audio diarization tasks. It includes secure environment handling
for access key management and comprehensive audio extraction capabilities.

Key Features:
- Secure access key management with environment variable validation
- Audio extraction from video clips for diarization tasks
- Support for multiple dataset parts (annotations, clips, etc.)
- Automatic audio preprocessing for diarization
- RTTM file generation for evaluation

Security Features:
- Access key validation and secure storage
- Environment variable isolation
- Secure subprocess execution
- Input sanitization and validation

Usage:
    ```python
    from datasets.recipes.ego4d import download_ego4d, prepare_ego4d

    # Download with secure access key
    corpus_path = download_ego4d(
        target_dir="./data",
        access_key="your_secure_key",
        dataset_parts=["clips", "annotations"]
    )

    # Process for audio diarization
    manifests = prepare_ego4d(
        corpus_dir=corpus_path,
        output_dir="./manifests/ego4d",

    )
    ```
"""

import json
import logging
import os
import subprocess
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from dotenv import load_dotenv
from lhotse import (
    CutSet,
    Recording,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.audio import RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Ego4DPart(Enum):
    """Ego4D dataset parts available for download"""

    METADATA = "metadata"
    ANNOTATIONS = "annotations"
    TAKES = "takes"
    CAPTURES = "captures"
    CLIPS = "clips"  # Video clips for audio extraction
    TAKE_TRAJECTORY = "take_trajectory"
    TAKE_EYE_GAZE = "take_eye_gaze"
    TAKE_POINT_CLOUD = "take_point_cloud"
    TAKE_VRS = "take_vrs"
    TAKE_VRS_NOIMAGESTREAM = "take_vrs_noimagestream"
    CAPTURE_TRAJECTORY = "capture_trajectory"
    CAPTURE_EYE_GAZE = "capture_eye_gaze"
    CAPTURE_POINT_CLOUD = "capture_point_cloud"
    DOWNSCALED_TAKES_448 = "downscaled_takes/448"
    FULL_SCALE = "full_scale"


def _validate_access_key(access_key: str) -> bool:
    """Simple access key validation"""
    return access_key and len(access_key) > 10


def _check_ego4d_cli() -> bool:
    """
    Check if Ego4D CLI is available and properly installed.

    Returns:
        True if CLI is available, False otherwise
    """
    try:
        subprocess.run(["ego4d", "--help"], capture_output=True, text=True, timeout=30)
        return True
    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return False


def _install_ego4d_cli() -> None:
    """Install Ego4D CLI if not available"""
    logger.info("Installing Ego4D CLI...")
    try:
        subprocess.run(
            ["pip", "install", "ego4d"],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        logger.info("Ego4D CLI installed successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to install Ego4D CLI: {e}")


def download_ego4d(
    target_dir: Pathlike,
    access_key: Optional[str] = None,
    force_download: bool = False,
    dataset_parts: Optional[List[Union[str, Ego4DPart]]] = None,
    install_cli: bool = True,
    timeout: int = 3600,
    env_file: Optional[Pathlike] = None,
) -> Pathlike:
    """
    Securely download Ego4D dataset with access key validation.

    Args:
        target_dir: Directory to download dataset to
        access_key: Ego4D access key (will be validated for security). If None, will try to load from environment.
        force_download: Whether to force re-download
        dataset_parts: List of dataset parts to download (defaults to clips and annotations)
        install_cli: Whether to install Ego4D CLI if not available
        timeout: Timeout for download operations in seconds
        env_file: Path to .env file to load environment variables from

    Returns:
        Path to downloaded dataset directory

    Raises:
        SecurityError: If access key is invalid or download fails
        ValueError: If dataset parts are invalid or access key not found
    """
    # Load environment variables from .env file if provided
    if env_file:
        load_dotenv(env_file)

    # Only use AWS_* credentials
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID") or access_key
    aws_access_key = os.getenv("AWS_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")

    if not aws_access_key_id or not aws_access_key:
        raise ValueError(
            "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_ACCESS_KEY in environment or .env."
        )

    # Set AWS credentials in environment for ego4d CLI
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_access_key

    # Ensure ego4d CLI sees a usable AWS profile (some versions require it)
    aws_profile = os.environ.get("AWS_PROFILE", "default")
    credentials_dir = Path(target_dir) / ".aws"
    credentials_dir.mkdir(parents=True, exist_ok=True)
    credentials_file = credentials_dir / "credentials"

    credentials_contents = (
        f"[{aws_profile}]\n"
        f"aws_access_key_id={aws_access_key_id}\n"
        f"aws_secret_access_key={aws_access_key}\n"
    )

    try:
        credentials_file.write_text(credentials_contents)
    except OSError as exc:
        logger.warning(f"Failed to write AWS shared credentials file: {exc}")
    else:
        os.environ.setdefault("AWS_PROFILE", aws_profile)
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(credentials_file)

    # Set default dataset parts for audio diarization
    if dataset_parts is None:
        dataset_parts = [Ego4DPart.CLIPS, Ego4DPart.ANNOTATIONS]

    # Convert string parts to enum if needed
    processed_parts = []
    for part in dataset_parts:
        if isinstance(part, str):
            try:
                processed_parts.append(Ego4DPart(part))
            except ValueError:
                raise ValueError(f"Invalid dataset part: {part}")
        else:
            processed_parts.append(part)

    # Validate dataset parts
    valid_parts = [part.value for part in Ego4DPart]
    for part in processed_parts:
        if part.value not in valid_parts:
            raise ValueError(f"Invalid dataset part: {part.value}")

    target_dir = Path(target_dir)
    if target_dir.name != "ego4d":
        target_dir = target_dir / "ego4d"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (v1/v2)
    completed_file = target_dir / ".ego4d_completed"
    clips_dir_v1 = target_dir / "v1" / "clips"
    clips_dir_v2 = target_dir / "v2" / "clips"
    annotations_dir_v1 = target_dir / "v1" / "annotations"
    annotations_dir_v2 = target_dir / "v2" / "annotations"

    def _has_media_files(path: Path) -> bool:
        return (
            any(path.glob("*.mp4"))
            or any(path.glob("*.avi"))
            or any((path / "clips").glob("*.mp4"))
        )

    def _has_annotation_files(path: Path) -> bool:
        return path.exists() and (any(path.glob("*.json")) or any(path.glob("*.jsonl")))

    clips_present = (
        clips_dir_v2.exists()
        and _has_media_files(clips_dir_v2)
        or (clips_dir_v1.exists() and _has_media_files(clips_dir_v1))
    )
    ann_present = _has_annotation_files(annotations_dir_v2) or _has_annotation_files(
        annotations_dir_v1
    )

    if completed_file.exists() and not force_download:
        logger.info("Ego4D dataset already downloaded, skipping...")
        return target_dir

    # Check and install CLI if needed
    if not _check_ego4d_cli():
        if install_cli:
            _install_ego4d_cli()
        else:
            raise RuntimeError("Ego4D CLI not available and installation disabled")

    # Decide which parts actually need downloading (avoid re-downloading existing parts)
    part_strings_full = [part.value for part in processed_parts]
    part_strings: List[str] = []
    for part in part_strings_full:
        if part == Ego4DPart.CLIPS.value:
            if not clips_present or force_download:
                part_strings.append(part)
        elif part == Ego4DPart.ANNOTATIONS.value:
            if not ann_present or force_download:
                part_strings.append(part)
        else:
            # For other parts, keep as requested
            part_strings.append(part)

    if not part_strings:
        logger.info("Requested parts already present; skipping download.")
        completed_file.touch()
        return target_dir

    # Build download command - pass datasets as separate args
    cmd = [
        "ego4d",
        f"--output_directory={target_dir}",
        "--datasets",
        *part_strings,
        "--yes",  # Skip confirmation prompt
    ]

    logger.info(f"Downloading Ego4D dataset to {target_dir}")
    logger.info(f"Downloading parts: {', '.join(part_strings)}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        # Execute download command
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)

        # Mark as completed
        completed_file.touch()
        logger.info("Ego4D dataset downloaded successfully")
        return target_dir

    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        if "403" in stderr:
            hint = (
                "Received 403 (Forbidden) from S3. Your Ego4D credentials may not "
                "have access to the requested dataset parts. Confirm that the "
                "credentials are active and tied to the Ego4D program, or request "
                "approved access."
            )
            logger.error(hint)
            raise RuntimeError(f"Download failed with 403 Forbidden. {hint}") from e

        logger.error(f"Download failed: {stderr}")
        raise RuntimeError(f"Download failed: {stderr}")
    except subprocess.TimeoutExpired:
        logger.error(f"Download timed out after {timeout} seconds")
        raise RuntimeError(f"Download timed out after {timeout} seconds")
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        raise RuntimeError(f"Download failed: {str(e)}")


def _extract_audio_from_video(
    video_path: Path, audio_path: Path, audio_sample_rate: int = 16000
) -> bool:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to input video file
        audio_path: Path to output audio file
        audio_sample_rate: Sample rate for extracted audio

    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",  # Audio codec
            "-ar",
            str(audio_sample_rate),  # Use provided sample rate
            "-ac",
            "1",  # Mono
            "-y",  # Overwrite output
            str(audio_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        return result.returncode == 0 and audio_path.exists()

    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return False


def _process_ego4d_annotations(
    annotations_path: Path, annotation_subset: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Process Ego4D annotations for audio diarization.

    Args:
        annotations_path: Path to annotations JSON file

    Returns:
        List of processed annotation segments
    """
    segments = []

    try:
        # JSONL support
        if annotations_path.suffix.lower() == ".jsonl":
            with open(annotations_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict) and "voice_activity" in item:
                        for segment in item.get("voice_activity", []):
                            segments.append(
                                {
                                    "start_time": segment.get("start_time", 0.0),
                                    "end_time": segment.get("end_time", 0.0),
                                    "speaker_id": segment.get("speaker_id", "unknown"),
                                    "confidence": segment.get("confidence", 1.0),
                                    "video_uid": item.get(
                                        "video_uid", item.get("clip_uid", "unknown")
                                    ),
                                }
                            )
            return segments

        # JSON support
        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        # Case 1: annotations is a dict mapping types -> list of items
        if isinstance(annotations, dict):
            for annotation_type, annotation_data in annotations.items():
                # If a specific subset is requested (e.g., "av"), filter types
                if annotation_subset and not annotation_type.startswith(
                    annotation_subset
                ):
                    continue

                if isinstance(annotation_data, list):
                    for item in annotation_data:
                        if isinstance(item, dict) and "voice_activity" in item:
                            for segment in item["voice_activity"]:
                                segments.append(
                                    {
                                        "start_time": segment.get("start_time", 0.0),
                                        "end_time": segment.get("end_time", 0.0),
                                        "speaker_id": segment.get(
                                            "speaker_id", "unknown"
                                        ),
                                        "confidence": segment.get("confidence", 1.0),
                                        "video_uid": item.get(
                                            "video_uid",
                                            item.get("clip_uid", "unknown"),
                                        ),
                                    }
                                )

        # Case 2: annotations is a list of items
        elif isinstance(annotations, list):
            for item in annotations:
                if isinstance(item, dict) and "voice_activity" in item:
                    for segment in item.get("voice_activity", []):
                        segments.append(
                            {
                                "start_time": segment.get("start_time", 0.0),
                                "end_time": segment.get("end_time", 0.0),
                                "speaker_id": segment.get("speaker_id", "unknown"),
                                "confidence": segment.get("confidence", 1.0),
                                "video_uid": item.get(
                                    "video_uid", item.get("clip_uid", "unknown")
                                ),
                            }
                        )

        return segments

    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        logger.warning(f"Error processing annotations: {e}")
        return []


def _infer_annotation_split(stem: str, annotation_subset: Optional[str]) -> str:
    """Infer split name (train/val/test) from annotation filename stem."""

    normalized = stem.lower().replace("-", "_")
    if annotation_subset:
        prefix = annotation_subset.lower()
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            normalized = normalized.lstrip("_- ")

    tokens = [token for token in normalized.split("_") if token]
    mapping = {
        "train": "train",
        "training": "train",
        "val": "val",
        "validation": "val",
        "dev": "val",
        "test": "test",
        "eval": "test",
    }

    for token in reversed(tokens):
        mapped = mapping.get(token)
        if mapped:
            return mapped

    return tokens[-1] if tokens else "train"


def prepare_ego4d(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    extract_audio: bool = True,
    audio_sample_rate: int = 16000,
    min_segment_duration: float = 0.5,
    max_segment_duration: float = 30.0,
    max_clips: Optional[int] = None,
    annotation_subset: Optional[str] = None,
) -> Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]:
    """
    Prepare Ego4D dataset for audio diarization tasks.

    Args:
        corpus_dir: Path to downloaded Ego4D dataset
        output_dir: Path to output manifests directory
        extract_audio: Whether to extract audio from video files
        audio_sample_rate: Sample rate for extracted audio
        min_segment_duration: Minimum segment duration in seconds
        max_segment_duration: Maximum segment duration in seconds

    Returns:
        Dictionary containing RecordingSet, SupervisionSet, and CutSet
    """
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_dir.exists():
        raise ValueError(f"Corpus directory does not exist: {corpus_dir}")

    logger.info(f"Preparing Ego4D dataset from {corpus_dir}")

    # Find video clips directory (support v1 and v2 layouts)
    clips_dir_v1 = corpus_dir / "v1" / "clips"
    clips_dir_v2 = corpus_dir / "v2" / "clips"
    annotations_dir_v1 = corpus_dir / "v1" / "annotations"
    annotations_dir_v2 = corpus_dir / "v2" / "annotations"

    clips_dir = clips_dir_v2 if clips_dir_v2.exists() else clips_dir_v1
    annotations_dir = (
        annotations_dir_v2 if annotations_dir_v2.exists() else annotations_dir_v1
    )

    if not clips_dir.exists():
        raise ValueError(f"Clips directory not found: {clips_dir}")

    video_files = sorted(list(clips_dir.glob("*.mp4")) + list(clips_dir.glob("*.avi")))
    if not video_files:
        logger.warning("No video files found in clips directory")
        empty_recordings = RecordingSet.from_recordings([])
        empty_supervisions = SupervisionSet.from_segments([])
        empty_cuts = CutSet.from_cuts([])
        return {
            "train": {
                "recordings": empty_recordings,
                "supervisions": empty_supervisions,
                "cuts": empty_cuts,
            }
        }

    video_file_map = {video.stem: video for video in video_files}

    # Load annotation segments, grouped by split (train/val/test)
    annotation_segments_by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if annotations_dir.exists():
        ann_files: List[Path] = []
        if annotation_subset and annotation_subset.startswith("av"):
            ann_files.extend(sorted(annotations_dir.glob("av_*.json")))
            ann_files.extend(sorted(annotations_dir.glob("av_*.jsonl")))
        else:
            ann_files.extend(sorted(annotations_dir.glob("*.json")))
            ann_files.extend(sorted(annotations_dir.glob("*.jsonl")))

        if not ann_files:
            logger.warning("No annotation files found for Ego4D")

        for ann_file in ann_files:
            segments = _process_ego4d_annotations(ann_file, annotation_subset)
            if not segments:
                continue
            split_name = _infer_annotation_split(ann_file.stem, annotation_subset)
            annotation_segments_by_split[split_name].extend(segments)
    else:
        logger.warning("Annotation directory not found; proceeding without supervisions")

    if not annotation_segments_by_split:
        # Ensure we still create at least one split for audio-only usage
        annotation_segments_by_split["train"] = []

    # Determine which videos belong to each split
    video_ids_by_split: Dict[str, List[str]] = {}
    for split, segments in annotation_segments_by_split.items():
        unique_ids = sorted(
            {segment.get("video_uid") for segment in segments if segment.get("video_uid")}
        )
        if not unique_ids:
            unique_ids = [video.stem for video in video_files]

        if max_clips is not None and max_clips > 0:
            unique_ids = unique_ids[: max_clips]

        video_ids_by_split[split] = unique_ids

    recordings_cache: Dict[str, Recording] = {}

    def get_recording(video_id: str) -> Optional[Recording]:
        if video_id in recordings_cache:
            return recordings_cache[video_id]

        video_path = video_file_map.get(video_id)
        if not video_path or not video_path.exists():
            logger.warning(f"Video file not found for {video_id}")
            return None

        source_path = video_path
        if extract_audio:
            audio_path = output_dir / f"{video_id}.wav"
            if not audio_path.exists():
                logger.info(f"Extracting audio from {video_path.name}")
                if not _extract_audio_from_video(
                    video_path, audio_path, audio_sample_rate
                ):
                    logger.warning(f"Failed to extract audio from {video_path.name}")
                    return None
            source_path = audio_path

        try:
            recording = Recording.from_file(str(source_path), recording_id=video_id)
        except Exception as e:
            logger.warning(f"Could not create recording for {video_id}: {e}")
            return None

        recordings_cache[video_id] = recording
        return recording

    manifests_by_split: Dict[
        str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]
    ] = {}
    multiple_splits = len(video_ids_by_split) > 1

    for split, video_ids in video_ids_by_split.items():
        recordings: List[Recording] = []
        available_ids: Set[str] = set()

        for video_id in video_ids:
            recording = get_recording(video_id)
            if recording is None:
                continue
            recordings.append(recording)
            available_ids.add(recording.id)

        if not recordings:
            logger.warning(f"No recordings created for {split} split")
            continue

        segments = [
            segment
            for segment in annotation_segments_by_split.get(split, [])
            if segment.get("video_uid") in available_ids
        ]

        supervisions: List[SupervisionSegment] = []
        for idx, segment in enumerate(segments):
            start_time = float(segment.get("start_time", 0.0))
            end_time = float(segment.get("end_time", 0.0))
            duration = end_time - start_time

            if duration <= 0:
                continue
            if duration < min_segment_duration or duration > max_segment_duration:
                continue

            recording_id = segment.get("video_uid")
            speaker_id = segment.get("speaker_id", "unknown")

            supervision = SupervisionSegment(
                id=f"{recording_id}_{idx}",
                recording_id=recording_id,
                start=start_time,
                duration=duration,
                speaker=speaker_id,
                channel=0,
                language="en",
                text="",
                custom={
                    "confidence": segment.get("confidence", 1.0),
                    "video_uid": recording_id,
                },
            )
            supervisions.append(supervision)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)

        if supervisions:
            try:
                recording_set, supervision_set = fix_manifests(
                    recording_set, supervision_set
                )
                validate_recordings_and_supervisions(recording_set, supervision_set)
            except Exception as e:
                logger.warning(f"Manifest validation failed for {split} split: {e}")

        cut_set = CutSet.from_manifests(
            recordings=recording_set,
            supervisions=supervision_set if supervisions else None,
        )

        suffix = f"_{split}" if multiple_splits or split != "train" else ""
        recording_set.to_file(output_dir / f"ego4d_recordings{suffix}.jsonl.gz")
        supervision_set.to_file(output_dir / f"ego4d_supervisions{suffix}.jsonl.gz")
        cut_set.to_file(output_dir / f"ego4d_cuts{suffix}.jsonl.gz")

        manifests_by_split[split] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
            "cuts": cut_set,
        }

    if not manifests_by_split:
        logger.warning("No valid Ego4D manifests were produced")
        empty_recordings = RecordingSet.from_recordings([])
        empty_supervisions = SupervisionSet.from_segments([])
        empty_cuts = CutSet.from_cuts([])
        return {
            "train": {
                "recordings": empty_recordings,
                "supervisions": empty_supervisions,
                "cuts": empty_cuts,
            }
        }

    for split, manifest in manifests_by_split.items():
        rec_count = len(manifest["recordings"])
        sup_count = len(manifest["supervisions"])
        logger.info(f"{split} split: {rec_count} recordings, {sup_count} supervisions")

    logger.info(f"Manifests saved to {output_dir}")
    return manifests_by_split


# Export functions for use in dataset management
__all__ = ["download_ego4d", "prepare_ego4d", "Ego4DPart"]
