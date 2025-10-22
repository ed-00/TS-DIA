"""Shared audio utility functions for recipes.

Provides helpers to detect sample rate and resample WAV files using ffmpeg/ffprobe
in a consistent, reusable way. Recipes should import these functions instead of
defining nested helpers.
"""
from __future__ import annotations

import json
import logging
import subprocess
import wave
from pathlib import Path
from typing import Optional


def _probe_sample_rate(path: Path) -> Optional[int]:
    """Try to get the sample rate of a file using wave header or ffprobe."""
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getframerate()
    except Exception:
        pass

    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            str(path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode == 0 and res.stdout:
            info = json.loads(res.stdout)
            streams = info.get("streams", [])
            if streams:
                sr = streams[0].get("sample_rate")
                if sr:
                    return int(sr)
    except Exception:
        pass

    return None


def resample_if_needed(in_path: Path, target_sr: int) -> Path:
    """Return path to file at target_sr. If already correct, return original path.

    If resampling is required, create a new file with suffix `_sr{target}` next to original
    and return its path. Falls back to original on failure.
    """
    sr = _probe_sample_rate(in_path)
    needs_resample = sr is None or sr != target_sr

    if not needs_resample:
        return in_path

    out_path = in_path.parent / f"{in_path.stem}_sr{target_sr}.wav"
    try:
        cmd = [
            "ffmpeg",
            "-i",
            str(in_path),
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(target_sr),
            "-ac",
            "1",
            "-y",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        if out_path.exists():
            return out_path
    except Exception as e:
        logging.warning(f"Resampling failed for {in_path}: {e}")

    return in_path


def resample_dir(directory: Path, target_sr: int) -> None:
    """Recursively resample all .wav files under `directory` to `target_sr`.

    Writes resampled files with suffix `_sr{target}` next to original files.
    """
    if not directory.exists():
        return
    for wav in directory.rglob("*.wav"):
        try:
            resample_if_needed(wav, target_sr)
        except Exception as e:
            logging.warning(f"Failed to resample {wav}: {e}")
