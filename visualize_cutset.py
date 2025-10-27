#! /usr/bin/env python
"""Create a video visualizing CutSet features and annotations for inspection.

Example usage (AVA-AVD, recording _a9SWtcaNj8):

    python visualize_cutset.py \
        --manifests-dir outputs/manifests/ava_avd \
        --recording-id _a9SWtcaNj8 \
        --output ava_avd__a9SWtcaNj8.mp4 \
        --window 30 \
        --step 2 \
        --fps 10

This produces an MP4 where the top panel shows a log-mel spectrogram window
sliding across the recording, and the bottom panel overlays diarization
segments (speakers) within the same window.
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

try:
    import torch
    import torchaudio
except ImportError as exc:  # pragma: no cover - torchaudio should be available
    raise RuntimeError(
        "torchaudio is required for visualize_cutset.py. Install torchaudio first."
    ) from exc

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional dependency
    sf = None


def write_audio(path: Path, tensor: torch.Tensor, sample_rate: int) -> None:
    """Save audio tensor (channels, samples) to WAV."""
    array = tensor.cpu().numpy().astype(np.float32)
    if sf is not None:
        sf.write(str(path), array.T, sample_rate)
        return

    data = np.clip(array, -1.0, 1.0)
    pcm = (data * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm.shape[0])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.T.tobytes())

from lhotse import RecordingSet, SupervisionSegment, SupervisionSet



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an animated spectrogram with diarization overlays from manifests."
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        required=True,
        help="Directory containing recordings_*.jsonl.gz and supervisions_*.jsonl.gz files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name to load (train|val|test). Defaults to 'train'.",
    )
    parser.add_argument(
        "--recording-id",
        type=str,
        required=True,
        help="Recording ID to visualize (e.g., _a9SWtcaNj8).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output MP4 path (will be overwritten).",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time (seconds) for the visualization window (default: 0).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration (seconds) to visualize. Defaults to full recording length.",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=20.0,
        help="Sliding window length in seconds shown per frame (default: 20).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=2.0,
        help="Advance step between frames in seconds (default: 2).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frames per second. Default chooses real-time playback (frames/duration).",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=40,
        help="Number of Mel bins when computing spectrogram (default: 40).",
    )
    parser.add_argument(
        "--frame-length",
        type=float,
        default=0.025,
        help="Frame length for STFT/Mel (seconds, default: 0.025).",
    )
    parser.add_argument(
        "--frame-shift",
        type=float,
        default=0.01,
        help="Frame shift / hop length (seconds, default: 0.01).",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Optional speed-up factor applied to playback (scales FPS).",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip embedding audio into the output video.",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=0,
        help="Resample audio to this rate before muxing (0 keeps/auto-selects original).",
    )
    parser.add_argument(
        "--audio-gain",
        type=float,
        default=1.0,
        help="Multiply audio by this factor before muxing (default: 1.0).",
    )
    parser.add_argument(
        "--audio-bitrate",
        type=int,
        default=128,
        help="Audio bitrate in kbps for the encoded track (default: 128).",
    )
    parser.add_argument(
        "--force-stereo",
        action="store_true",
        help="Duplicate mono audio to stereo before encoding (helps picky players).",
    )
    parser.add_argument(
        "--output-video-fps",
        type=float,
        default=30.0,
        help="FFmpeg output FPS when re-encoding the spectrogram video (default: 30).",
    )
    parser.add_argument(
        "--video-crf",
        type=float,
        default=20.0,
        help="CRF to use with libx264 when re-encoding the video (default: 20).",
    )
    parser.add_argument(
        "--video-preset",
        type=str,
        default="veryfast",
        help="FFmpeg preset for libx264 (default: veryfast).",
    )
    return parser.parse_args()


def load_manifests(manifests_dir: Path, split: str) -> tuple[RecordingSet, SupervisionSet]:
    recordings_path = manifests_dir / f"ava_avd_recordings_{split}.jsonl.gz"
    supervisions_path = manifests_dir / f"ava_avd_supervisions_{split}.jsonl.gz"
    if not recordings_path.exists():
        raise FileNotFoundError(f"Recordings manifest not found: {recordings_path}")
    if not supervisions_path.exists():
        raise FileNotFoundError(f"Supervisions manifest not found: {supervisions_path}")
    recordings = RecordingSet.from_file(recordings_path)
    supervisions = SupervisionSet.from_file(supervisions_path)
    return recordings, supervisions


def compute_log_mel(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mels: int,
    frame_length: float,
    frame_shift: float,
) -> np.ndarray:
    """Compute log-mel spectrogram from mono waveform."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # downmix to mono

    n_fft = int(round(sample_rate * frame_length))
    win_length = n_fft
    hop_length = int(round(sample_rate * frame_shift))

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
        center=True,
    )
    mel = mel_transform(waveform)
    log_mel = torch.log(mel + 1e-6)
    return log_mel.squeeze(0).numpy()


def collect_speakers(supervisions: Sequence[SupervisionSegment]) -> List[str]:
    speakers = sorted({seg.speaker or "unknown" for seg in supervisions})
    return speakers


def filter_supervisions(
    supervisions: Iterable[SupervisionSegment],
    start: float,
    end: float,
) -> List[SupervisionSegment]:
    result = []
    for seg in supervisions:
        seg_end = seg.start + seg.duration
        if seg_end < start or seg.start > end:
            continue
        result.append(seg)
    return result


def prepare_output_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int | None,
    gain: float,
    force_stereo: bool,
) -> tuple[torch.Tensor, int]:
    """Optionally resample, amplify, and upmix audio before muxing."""
    target_sr = sample_rate if target_sample_rate in (None, sample_rate) else target_sample_rate
    output = waveform
    if target_sr != sample_rate:
        print(f"Resampling audio for output: {sample_rate} Hz -> {target_sr} Hz")
        output = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    if force_stereo and output.shape[0] == 1:
        print("Duplicating mono audio to stereo for compatibility")
        output = output.repeat(2, 1)
    if not math.isclose(gain, 1.0, rel_tol=1e-6):
        print(f"Applying audio gain {gain:.2f}x before muxing")
        output = torch.clamp(output * gain, -1.0, 1.0)
    return output, target_sr


def main() -> None:
    args = parse_args()

    recordings, supervisions = load_manifests(args.manifests_dir, args.split)
    if getattr(recordings, "is_lazy", False):
        recordings = recordings.to_eager()
    if getattr(supervisions, "is_lazy", False):
        supervisions = supervisions.to_eager()

    if args.recording_id not in recordings:
        raise ValueError(f"Recording ID {args.recording_id} not found in manifests.")

    recording = recordings[args.recording_id]
    sample_rate = int(recording.sampling_rate)

    start = max(0.0, args.start)
    full_duration = recording.duration
    if args.duration is None:
        duration = full_duration - start
    else:
        duration = min(args.duration, full_duration - start)
    if duration <= 0:
        raise ValueError("Requested duration is non-positive after bounds checking.")

    print(f"Loading audio {args.recording_id} [{start:.2f}, {start + duration:.2f}]...")
    audio = recording.load_audio()
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    sample_start = int(round(start * sample_rate))
    sample_end = int(round((start + duration) * sample_rate))
    audio = audio[:, sample_start:sample_end]
    waveform = torch.from_numpy(audio).float()

    print("Computing log-mel spectrogram...")
    log_mel = compute_log_mel(
        waveform,
        sample_rate=sample_rate,
        n_mels=args.n_mels,
        frame_length=args.frame_length,
        frame_shift=args.frame_shift,
    )
    frame_shift = args.frame_shift
    num_frames = log_mel.shape[1]
    time_axis = np.arange(num_frames) * frame_shift + start
    total_window = args.window
    step = args.step
    if total_window <= 0 or step <= 0:
        raise ValueError("Window and step must be positive.")

    relevant_supervisions = [
        seg
        for seg in supervisions
        if seg.recording_id == args.recording_id
        and seg.start + seg.duration > start
        and seg.start < start + duration
    ]
    speakers = collect_speakers(relevant_supervisions)
    speaker_to_index = {spk: idx for idx, spk in enumerate(speakers)}

    cm = plt.get_cmap("tab20")
    colors = {spk: cm(idx % cm.N) for spk, idx in speaker_to_index.items()}

    # Precompute normalization for color scale to keep animation stable.
    vmin = float(np.percentile(log_mel, 1))
    vmax = float(np.percentile(log_mel, 99))

    frames_count = max(1, math.ceil((duration - total_window) / step) + 1)
    fps_info = args.fps if args.fps is not None else "auto"
    print(
        f"Preparing animation: window={total_window}s, step={step}s, frames={frames_count}, fps={fps_info}"
    )

    fig, (ax_spec, ax_sup) = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    fig.suptitle(f"Recording {args.recording_id}")

    # Initial slice
    def extract_slice(start_time: float) -> tuple[np.ndarray, float, float]:
        end_time = min(start_time + total_window, start + duration)
        start_idx = max(0, int((start_time - start) / frame_shift))
        end_idx = min(num_frames, int((end_time - start) / frame_shift))
        slice_data = log_mel[:, start_idx:end_idx]
        if slice_data.size == 0:
            slice_data = np.zeros((log_mel.shape[0], 1), dtype=np.float32)
        return slice_data, start_time, end_time

    initial_slice, slice_start, slice_end = extract_slice(start)
    extent = [slice_start, slice_end, 0, args.n_mels]
    spec_im = ax_spec.imshow(
        initial_slice,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
    )
    ax_spec.set_ylabel("Mel bins")
    fig.colorbar(spec_im, ax=ax_spec, pad=0.01, label="Log energy")

    if speakers:
        ax_sup.set_ylim(-0.5, len(speakers) - 0.5)
        ax_sup.set_yticks(range(len(speakers)))
        ax_sup.set_yticklabels(speakers)
    else:
        ax_sup.set_ylim(-0.5, 0.5)
        ax_sup.set_yticks([0])
        ax_sup.set_yticklabels(["no speakers"])
    ax_sup.set_xlabel("Time (s)")

    def draw_supervisions(current_start: float, current_end: float) -> None:
        ax_sup.clear()
        ax_sup.set_xlim(current_start, current_end)
        if speakers:
            ax_sup.set_ylim(-0.5, len(speakers) - 0.5)
            ax_sup.set_yticks(range(len(speakers)))
            ax_sup.set_yticklabels(speakers)
        else:
            ax_sup.set_ylim(-0.5, 0.5)
            ax_sup.set_yticks([0])
            ax_sup.set_yticklabels(["no speakers"])
        ax_sup.set_xlabel("Time (s)")

        visible = filter_supervisions(relevant_supervisions, current_start, current_end)
        for seg in visible:
            seg_start = max(seg.start, current_start)
            seg_end = min(seg.start + seg.duration, current_end)
            width = max(1e-3, seg_end - seg_start)
            spk = seg.speaker or "unknown"
            y = speaker_to_index.get(spk, 0)
            ax_sup.broken_barh(
                [(seg_start, width)],
                (y - 0.4, 0.8),
                facecolors=colors.get(spk, "#777777"),
                edgecolors="black",
                linewidth=0.3,
            )
        ax_sup.grid(True, axis="x", linestyle="--", alpha=0.4)

    draw_supervisions(slice_start, slice_end)

    def update(frame_idx: int):
        current_start = start + frame_idx * step
        current_start = min(current_start, start + duration - total_window)
        current_start = max(start, current_start)
        slice_data, slice_start_t, slice_end_t = extract_slice(current_start)
        spec_im.set_data(slice_data)
        spec_im.set_extent([slice_start_t, slice_end_t, 0, args.n_mels])
        ax_spec.set_xlim(slice_start_t, slice_end_t)
        draw_supervisions(slice_start_t, slice_end_t)
        return [spec_im]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames_count,
        blit=False,
        repeat=False,
    )

    base_fps = args.fps
    if base_fps is None or base_fps <= 0:
        base_fps = frames_count / duration if duration > 0 else 1.0
    playback_fps = max(base_fps * args.playback_speed, 1e-3)
    video_duration = frames_count / playback_fps if playback_fps > 0 else duration

    print(
        f"Rendering animation (frames={frames_count}, fps={playback_fps:.3f}, length≈{video_duration:.2f}s)..."
    )

    target_audio_sr = args.audio_sample_rate
    if target_audio_sr <= 0:
        if sample_rate >= 32000:
            target_audio_sr = sample_rate
        else:
            target_audio_sr = 48000
        if target_audio_sr != sample_rate:
            print(
                f"Auto-upsampling audio to {target_audio_sr} Hz for better player compatibility."
            )

    with tempfile.TemporaryDirectory(prefix="cutviz_") as tmpdir:
        tmp_dir = Path(tmpdir)
        temp_video = tmp_dir / "spectrogram.mp4"

        anim.save(
            str(temp_video),
            fps=playback_fps,
            dpi=150,
            writer="ffmpeg",
        )
        plt.close(fig)

        if args.no_audio:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_video), str(args.output))
            print(f"Saved video without audio to {args.output}")
            return

        temp_audio = tmp_dir / "audio.wav"
        output_waveform, output_sample_rate = prepare_output_audio(
            waveform,
            sample_rate,
            target_audio_sr,
            args.audio_gain,
            args.force_stereo,
        )
        write_audio(temp_audio, output_waveform, output_sample_rate)

        speed_factor = duration / video_duration if video_duration > 0 else 1.0

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_video),
            "-i",
            str(temp_audio),
        ]

        def factorize_atempo(factor: float) -> List[float]:
            components: List[float] = []
            if factor <= 0:
                return [1.0]
            remaining = factor
            while remaining < 0.5:
                components.append(0.5)
                remaining /= 0.5
            while remaining > 2.0:
                components.append(2.0)
                remaining /= 2.0
            if not math.isclose(remaining, 1.0, rel_tol=1e-3):
                components.append(remaining)
            return components

        audio_filters: List[str] = []
        if not math.isclose(speed_factor, 1.0, rel_tol=1e-3):
            tempo_filters = factorize_atempo(speed_factor)
            audio_filters.extend(
                f"atempo={value:.6f}" for value in tempo_filters if not math.isclose(value, 1.0, rel_tol=1e-3)
            )

        video_filters: List[str] = []
        if args.output_video_fps > 0:
            video_filters.append(f"fps={args.output_video_fps:.3f}")
        video_filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
        video_filters.append("format=yuv420p")

        ffmpeg_cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
        if video_filters:
            ffmpeg_cmd.extend(["-filter:v", ",".join(video_filters)])
        if audio_filters:
            ffmpeg_cmd.extend(["-filter:a", ",".join(audio_filters)])

        ffmpeg_cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                args.video_preset,
                "-crf",
                str(args.video_crf),
                "-c:a",
                "aac",
                "-b:a",
                f"{max(args.audio_bitrate, 32)}k",
            ]
        )
        if args.force_stereo:
            ffmpeg_cmd.extend(["-ac", "2"])

        ffmpeg_cmd.extend(["-movflags", "+faststart", "-shortest", str(args.output)])

        args.output.parent.mkdir(parents=True, exist_ok=True)
        if speed_factor > 4 or speed_factor < 0.25:
            print(
                f"Warning: audio speed factor {speed_factor:.2f} is extreme; consider reducing step or setting --fps."
            )
        print(
            "Combining audio with ffmpeg (speed factor {:.3f})...".format(speed_factor)
        )
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Saved video with audio to {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
