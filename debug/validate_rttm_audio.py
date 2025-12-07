#! /usr/bin/env python
"""Compare RTTM speech durations against extracted audio durations.

This utility helps spot cases where speech coverage is unexpectedly low by
summing diarization segments from RTTM files and comparing them with the
corresponding recording duration.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate RTTM-labelled speech coverage versus audio duration."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("./outputs/data/ava_avd/dataset"),
        help="Path containing dataset assets (rttms/, videos/, etc.).",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("./outputs/data/ava_avd/audio"),
        help="Directory with extracted audio WAV files (per base video id).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Optional split to filter (uses split/*.list entries).",
    )
    parser.add_argument(
        "--min-speech-ratio",
        type=float,
        default=0.0,
        help="Only show recordings with speech ratio below this threshold (0..1).",
    )
    return parser.parse_args()


def load_split_members(dataset_dir: Path, split_name: str) -> set[str]:
    if split_name == "all":
        return set()
    split_file = dataset_dir / "split" / f"{split_name}.list"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, "r", encoding="utf-8") as f:
        entries = {line.strip() for line in f if line.strip()}
    # Entries include chunk suffix _c_XX; we only need base ids
    return {entry.split("_c_")[0] for entry in entries}


def read_rttm_segments(rttm_path: Path) -> Iterable[Tuple[str, float, float]]:
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not line.upper().startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            start = float(parts[3])
            duration = float(parts[4])
            yield parts[1], start, duration


def collect_rttm_stats(rttm_dir: Path) -> Dict[str, float]:
    speech_durations: Dict[str, float] = defaultdict(float)
    for rttm_file in sorted(rttm_dir.glob("*.rttm")):
        base_id = rttm_file.stem.split("_c_")[0]
        for _, _, dur in read_rttm_segments(rttm_file):
            speech_durations[base_id] += dur
    return speech_durations


def ffprobe_duration(path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    try:
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def wav_duration(path: Path) -> float | None:
    try:
        import soundfile as sf

        with sf.SoundFile(path) as f:
            return f.frames / float(f.samplerate)
    except (ImportError, RuntimeError):
        pass
    try:
        import wave

        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return None


def resolve_audio_duration(audio_dir: Path, dataset_dir: Path, base_id: str) -> float | None:
    wav_path = audio_dir / f"{base_id}.wav"
    if wav_path.exists():
        return wav_duration(wav_path)
    # Fall back to video if WAV missing
    for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
        video_path = dataset_dir / "videos" / f"{base_id}{ext}"
        if video_path.exists():
            return ffprobe_duration(video_path)
    return None


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir
    rttm_dir = dataset_dir / "rttms"
    if not rttm_dir.exists():
        raise FileNotFoundError(f"RTTM directory not found: {rttm_dir}")

    split_filter = load_split_members(dataset_dir, args.split) if args.split != "all" else None
    speech_durations = collect_rttm_stats(rttm_dir)

    total_audio = 0.0
    total_speech = 0.0
    under_threshold = []

    for base_id, speech_dur in sorted(speech_durations.items()):
        if split_filter is not None and base_id not in split_filter:
            continue
        audio_dur = resolve_audio_duration(args.audio_dir, dataset_dir, base_id)
        if audio_dur is None:
            print(f"⚠️  {base_id}: audio file not found")
            continue
        total_audio += audio_dur
        total_speech += speech_dur
        ratio = speech_dur / audio_dur if audio_dur > 0 else 0.0
        if ratio <= args.min_speech_ratio:
            under_threshold.append((base_id, audio_dur, speech_dur, ratio))

    print("\n=== Dataset Summary ===")
    print(f"Total audio duration   : {total_audio / 3600:.2f} h")
    print(f"Total speech duration  : {total_speech / 3600:.2f} h")
    overall_ratio = total_speech / total_audio if total_audio > 0 else 0.0
    print(f"Overall speech ratio   : {overall_ratio * 100:.2f}%")

    threshold = args.min_speech_ratio
    print(f"\nRecordings with speech ratio <= {threshold * 100:.2f}%:")
    if not under_threshold:
        print("  (none)")
    else:
        for base_id, audio_dur, speech_dur, ratio in under_threshold:
            print(
                f"  {base_id:15s} audio={audio_dur/3600:.2f}h speech={speech_dur/3600:.2f}h "
                f"({ratio * 100:.2f}%)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
