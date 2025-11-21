# TS-DIA: Transformer-based Speaker Diarization

A transformer-based speaker diarization system for audio processing and analysis.

## Setup

### System Dependencies

This project requires FFmpeg for audio/video processing. Install it using:

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- `lhotse==1.31.1` - Audio data processing
- `yamlargparse==1.31.1` - YAML configuration parsing
- `gdown==5.2.0` - Google Drive file downloads
- `torchaudio>=2.0.0` - Audio processing with PyTorch

### Development Container

If using a development container, ensure FFmpeg is installed in the container. Add this to your Dockerfile or devcontainer.json:

```dockerfile
RUN apt-get update && apt-get install -y ffmpeg
```

## Usage

### Dataset Management

The system supports multiple datasets including AVA-AVD, MSWild, and VoxConverse. Use the data manager to download and prepare datasets:

```bash
python datasets/data_manager.py --config your_config.yml
```

### AVA-AVD Dataset

The AVA-AVD dataset requires video processing capabilities. Make sure FFmpeg is installed before processing:

```bash
# Test AVA-AVD integration (small subset)
python datasets/data_manager.py --config test_ava_avd_small.yml

# Full AVA-AVD processing (with videos)
python datasets/data_manager.py --config test_ava_avd_full.yml
```

## Performance Optimization

### Video Processing Speed

The AVA-AVD dataset processing has been optimized to avoid slow TorchAudio StreamReader calls:

- Uses `ffprobe` directly for metadata extraction (much faster)
- Avoids deprecated TorchAudio APIs that cause slowdowns
- Processes videos in parallel when possible

### Memory Usage

For large datasets:
- Consider processing splits separately
- Use `download_videos: false` for annotation-only testing
- Ensure sufficient disk space for video files

## Troubleshooting

### FFmpeg Issues

If you encounter FFmpeg-related errors:
1. Ensure FFmpeg is installed: `ffmpeg -version`
2. Check that the FFmpeg binary is in your PATH
3. For development containers, ensure FFmpeg is installed in the container

### TorchAudio Warnings

You may see deprecation warnings from TorchAudio. These are expected and don't affect functionality:
```
UserWarning: torio.io._streaming_media_decoder.StreamingMediaDecoder has been deprecated
```

### Video Processing

For video datasets like AVA-AVD:
- Ensure sufficient disk space (videos can be large)
- Consider using `download_videos: false` in config for testing without full video downloads
- Video processing requires significant memory and CPU resources

### CUDA Out of Memory (OOM) during attention

If you see a CUDA OOM originating from `causal_linear_attention` (e.g. an error pointing at `context_cumsum`), it means the causal linear attention implementation has built large intermediate tensors for the chunk being processed. Recommended steps:

- Reduce training/evaluation batch size in your config (fastest fix).
- Reduce `eval_knobs.sliding_window` or `global_config.features.batch_duration` to shorten sequence length.
- Reduce `model.encoder.nb_features` / `model.decoder.nb_features` to lower random-features dimensionality.
- Tune `causal_chunk_size` for the model layers — smaller chunk sizes lower peak memory at the cost of some runtime overhead.
- Run with mixed precision if not already enabled (config: `training.mixed_precision: true`).
- Set environment variable to avoid fragmentation:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Where possible it's recommended to try a smaller `causal_chunk_size` first. The code now supports passing `causal_chunk_size` through the model construction so you can set this from your model config (lower -> safer for memory).

## Project Structure

```
TS-DIA/
├── datasets/           # Dataset management and recipes
│   ├── recipes/       # Dataset-specific download/prepare functions
│   ├── data_manager.py # Main dataset management system
│   └── dataset_types.py # Dataset configuration classes
├── model/             # Transformer model components
├── data/              # Dataset storage
├── manifests/         # Processed dataset manifests
├── configs/           # Configuration files
└── examples/          # Usage examples
```

## Available Datasets

- **AVA-AVD**: Audio-visual diarization dataset with video files
- **MSWild**: Multi-speaker wild dataset
- **VoxConverse**: Speaker diarization benchmark dataset
