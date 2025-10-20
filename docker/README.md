# Docker Setup for TS-DIA Training

This directory contains Docker configurations for running TS-DIA training on any GPU configuration.

## Overview

The Docker setup uses the NVIDIA PyTorch base image (`nvcr.io/nvidia/pytorch:24.10-py3`) and provides flexible scripts for:
- Building the container image
- Running interactive sessions
- Running training directly
- Running distributed training with multiple GPUs

## Prerequisites

- Docker installed ([Install Docker](https://docs.docker.com/get-docker/))
- NVIDIA Docker runtime installed ([Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- NVIDIA GPU(s) available

### Verify GPU Access

```bash
# Check if Docker can access your GPUs
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## Quick Start

### 0. Setup (First Time Only)

Make the scripts executable:

```bash
cd /workspaces/TS-DIA

# Option 1: Use the setup script
chmod +x docker/setup_permissions.sh
./docker/setup_permissions.sh

# Option 2: Set permissions manually
chmod +x docker/*.sh
```

### 1. Build the Docker Image

From the project root directory:

```bash
# Build with default name (ts-dia-training)
./docker/build.sh

# Or specify a custom image name
./docker/build.sh my-custom-name

# Force rebuild if image already exists
./docker/build.sh ts-dia-training --force
```

**Note:** The build script automatically checks if the image already exists and skips building to save time. Use `--force` flag to rebuild.

### 2. Configure Storage Volumes (Optional)

Configure custom storage paths for fast, slow, and temp storage:

```bash
# Copy the example configuration
cp docker/volumes.env.example docker/volumes.env

# Edit with your storage paths
nano docker/volumes.env
```

**Example configuration:**
```bash
FAST_STORAGE=/mnt/nvme/ts-dia        # SSD/NVMe for checkpoints
SLOW_STORAGE=/mnt/hdd/datasets       # HDD for large datasets
TEMP_STORAGE=/tmp/ts-dia             # Temp/scratch space
```

**Note:** If not configured, default directories will be created in `./storage/` (works on any machine).

### 3. Run Training

Choose one of the following methods based on your needs:

#### Method A: CPU-Only for Data Preparation

Run data preparation and preprocessing without GPU:

```bash
# Interactive CPU-only session
./docker/run_cpu.sh

# Run specific data preparation command
./docker/run_cpu.sh ts-dia-training prep "python data_manager/data_manager.py --dataset voxconverse"
```

#### Method B: Interactive Container with GPU (Recommended for Development)

Run an interactive shell with access to your workspace:

```bash
# Use all available GPUs
./docker/run.sh

# Specify image name
./docker/run.sh ts-dia-training

# Specify image, container name, and specific GPUs
./docker/run.sh ts-dia-training my-session 0,1

# Inside the container, run training
python train.py --config configs/train/training_example.yml
```

#### Method C: Direct Training Execution

Run training directly without interactive shell:

```bash
# Basic usage (uses all GPUs)
./docker/run_training.sh configs/train/training_example.yml

# Specify all parameters
./docker/run_training.sh configs/train/full_experiment.yml ts-dia-training my-exp 0,1
```

#### Method D: Distributed Training with Accelerate

Run multi-GPU distributed training:

```bash
# Use all available GPUs (auto-detected, e.g., 8 GPUs)
./docker/run_distributed.sh configs/train/training_example.yml

# Specify number of GPUs (4 out of 8)
./docker/run_distributed.sh configs/train/full_experiment.yml 4

# Use all 8 GPUs explicitly
./docker/run_distributed.sh configs/train/encoder_model.yml 8 ts-dia-training my-ddp-exp

# Use custom Accelerate config for advanced distributed strategies (4 GPUs)
./docker/run_distributed.sh configs/train/encoder_model.yml 4 ts-dia-training my-exp configs/accelerate_fsdp.yaml

# Quick 2-GPU testing before scaling to full training
./docker/run_distributed.sh configs/train/training_example.yml 2 ts-dia-training test configs/accelerate_2gpu_test.yaml
```

## Detailed Usage

### Script: `build.sh`

Builds the Docker image with all required dependencies. Automatically checks if the image already exists and skips building unless forced.

**Usage:**
```bash
./docker/build.sh [IMAGE_NAME] [--force]
```

**Arguments:**
- `IMAGE_NAME` (optional): Name for the Docker image. Default: `ts-dia-training`
- `--force` (optional): Force rebuild even if image exists

**Examples:**
```bash
# Build with default name (skips if already exists)
./docker/build.sh

# Build with custom name
./docker/build.sh my-training-image

# Force rebuild even if image exists
./docker/build.sh ts-dia-training --force

# Force rebuild with custom name
./docker/build.sh my-custom-image --force
```

**Behavior:**
- First run: Builds the image from scratch
- Subsequent runs: Skips build if image exists (fast)
- With `--force`: Always rebuilds (useful after dependency changes)

---

### Script: `run.sh`

Starts an interactive container with the entire workspace mounted.

**Usage:**
```bash
./docker/run.sh [IMAGE_NAME] [CONTAINER_NAME] [GPU_IDS]
```

**Arguments:**
- `IMAGE_NAME` (optional): Docker image name. Default: `ts-dia-training`
- `CONTAINER_NAME` (optional): Container name. Default: `ts-dia-training-<timestamp>`
- `GPU_IDS` (optional): GPU devices to use. Default: `all`

**GPU Configuration Examples:**
```bash
# Use all available GPUs (e.g., 8 GPUs if you have 8)
./docker/run.sh ts-dia-training my-dev all

# Use only GPU 0
./docker/run.sh ts-dia-training my-dev 0

# Use GPU 0 and 1
./docker/run.sh ts-dia-training my-dev 0,1

# Use GPU 2, 3, and 5
./docker/run.sh ts-dia-training my-dev 2,3,5

# Use all 8 GPUs (explicit)
./docker/run.sh ts-dia-training my-dev 0,1,2,3,4,5,6,7
```

**Inside the Container:**
```bash
# Your workspace is mounted at /workspace
cd /workspace

# Run training
python train.py --config configs/train/training_example.yml

# Run with CLI overrides
python train.py --config configs/train/training_example.yml \
    --epochs 100 \
    --batch-size 64

# Access TensorBoard
tensorboard --logdir checkpoints/
```

---

### Script: `run_training.sh`

Runs training directly in a container without interactive mode.

**Usage:**
```bash
./docker/run_training.sh [CONFIG_FILE] [IMAGE_NAME] [CONTAINER_NAME] [GPU_IDS]
```

**Arguments:**
- `CONFIG_FILE` (optional): Path to config file. Default: `configs/train/training_example.yml`
- `IMAGE_NAME` (optional): Docker image name. Default: `ts-dia-training`
- `CONTAINER_NAME` (optional): Container name. Default: `ts-dia-training-<timestamp>`
- `GPU_IDS` (optional): GPU devices to use. Default: `all`

**Examples:**
```bash
# Train with default config on all GPUs
./docker/run_training.sh

# Train with specific config
./docker/run_training.sh configs/train/full_experiment.yml

# Train on specific GPUs
./docker/run_training.sh configs/train/encoder_model.yml ts-dia-training exp1 0,1,2,3
```

---

### Script: `run_cpu.sh`

Runs a CPU-only container for data preparation and preprocessing tasks (no GPU access).

**Usage:**
```bash
./docker/run_cpu.sh [IMAGE_NAME] [CONTAINER_NAME] [COMMAND]
```

**Arguments:**
- `IMAGE_NAME` (optional): Docker image name. Default: `ts-dia-training`
- `CONTAINER_NAME` (optional): Container name. Default: `ts-dia-cpu-<timestamp>`
- `COMMAND` (optional): Command to run. Default: `/bin/bash` (interactive)

**Examples:**
```bash
# Interactive CPU-only session
./docker/run_cpu.sh

# Named session
./docker/run_cpu.sh ts-dia-training my-data-prep

# Run data preparation command
./docker/run_cpu.sh ts-dia-training prep "python data_manager/data_manager.py --help"

# Process manifests
./docker/run_cpu.sh ts-dia-training prep "python -m data_manager.data_manager --dataset voxconverse --split dev"

# Extract features
./docker/run_cpu.sh ts-dia-training prep "python scripts/extract_features.py --config configs/feature_extraction.yml"
```

**Use Cases:**
- Data preprocessing and preparation
- Manifest generation
- Feature extraction (CPU-based)
- Dataset validation
- Configuration file generation
- Any task that doesn't require GPU

---

### Script: `run_distributed.sh`

Runs distributed training using Accelerate for multi-GPU training.

**Usage:**
```bash
./docker/run_distributed.sh [CONFIG_FILE] [NUM_GPUS] [IMAGE_NAME] [CONTAINER_NAME] [ACCELERATE_CONFIG]
```

**Arguments:**
- `CONFIG_FILE` (optional): Path to config file. Default: `configs/train/training_example.yml`
- `NUM_GPUS` (optional): Number of GPUs to use. Default: auto-detected (ignored if `ACCELERATE_CONFIG` is provided)
- `IMAGE_NAME` (optional): Docker image name. Default: `ts-dia-training`
- `CONTAINER_NAME` (optional): Container name. Default: `ts-dia-ddp-<timestamp>`
- `ACCELERATE_CONFIG` (optional): Path to Accelerate config file. If provided, overrides `NUM_GPUS`

**Examples:**
```bash
# Distributed training on all available GPUs (auto-detected, e.g., 8 GPUs)
./docker/run_distributed.sh configs/train/full_experiment.yml

# Distributed training on 4 GPUs (out of 8 available)
./docker/run_distributed.sh configs/train/encoder_model.yml 4

# Full 8-GPU training
./docker/run_distributed.sh configs/train/decoder_model.yml 8 ts-dia-training my-ddp

# Use custom Accelerate config file (4 GPUs)
./docker/run_distributed.sh configs/train/encoder_model.yml 4 ts-dia-training my-exp configs/accelerate_config.yaml

# Quick 2-GPU test with config file
./docker/run_distributed.sh configs/train/training_example.yml 2 ts-dia-training test configs/accelerate_2gpu_test.yaml

# With Accelerate config (NUM_GPUS is ignored, config controls GPU usage)
./docker/run_distributed.sh configs/train/full_experiment.yml "" ts-dia-training my-fsdp configs/accelerate_fsdp.yaml
```

**Accelerate Config:**
You can create a custom Accelerate configuration file to specify:
- Distributed strategy (DDP, FSDP, DeepSpeed)
- Mixed precision settings
- Gradient accumulation
- And more...

**Pre-configured Accelerate files:**
- `configs/accelerate_2gpu_test.yaml` - 2 GPU testing/debugging
- `configs/accelerate_ddp.yaml` - 8 GPU DDP training
- `configs/accelerate_fsdp.yaml` - 8 GPU FSDP for large models

Generate a custom Accelerate config:
```bash
# Run this in an interactive container
./docker/run.sh
accelerate config --config_file configs/accelerate_config.yaml
```

## Volume Mounting

### Workspace Volume

All scripts mount your **entire project directory** as `/workspace` inside the container. This means:

- All your code, configs, and data are accessible
- Changes made inside the container are reflected on your host
- Checkpoints, logs, and outputs are saved directly to your host filesystem

**Mounted as:** `-v $(pwd):/workspace`

### Storage Volumes (Fast, Slow, Temp)

The scripts support three additional storage volumes for flexible data management:

- **Fast Storage** (`/storage/fast`) - For checkpoints, models, and active data (SSD/NVMe)
- **Slow Storage** (`/storage/slow`) - For large datasets and archives (HDD/Network storage)
- **Temp Storage** (`/storage/temp`) - For temporary files and cache (Scratch space)

#### Configuration

1. **Copy the example configuration:**
   ```bash
   cp docker/volumes.env.example docker/volumes.env
   ```

2. **Edit `docker/volumes.env` with your storage paths:**
   ```bash
   # Example configuration
   FAST_STORAGE=/mnt/nvme/ts-dia
   SLOW_STORAGE=/mnt/hdd/datasets
   TEMP_STORAGE=/tmp/ts-dia
   ```

3. **Run containers (volumes are automatically mounted):**
   ```bash
   ./docker/run.sh
   ```

#### Default Behavior

If `docker/volumes.env` is not configured, the scripts will automatically create and use:
- `./storage/fast` for fast storage
- `./storage/slow` for slow storage
- `./storage/temp` for temp storage

This ensures the scripts work on any machine without configuration.

#### Using Storage Volumes in Your Code

Inside the container, access the volumes at:
```python
# Example in Python
FAST_STORAGE = "/storage/fast"
SLOW_STORAGE = "/storage/slow"
TEMP_STORAGE = "/storage/temp"

# Save checkpoints to fast storage
checkpoint_dir = f"{FAST_STORAGE}/checkpoints"

# Load datasets from slow storage
dataset_path = f"{SLOW_STORAGE}/voxconverse"

# Use temp storage for caching
cache_dir = f"{TEMP_STORAGE}/cache"
```

Or in your config files:
```yaml
checkpoint:
  save_dir: /storage/fast/checkpoints/my-experiment

dataset:
  data_path: /storage/slow/datasets/voxconverse

cache:
  dir: /storage/temp/cache
```

## Port Forwarding

The following ports are forwarded from the container to your host:

- **6006**: TensorBoard
- **8888**: Jupyter Notebook (if needed)

**Access TensorBoard:**
```bash
# Inside container
tensorboard --logdir checkpoints/ --host 0.0.0.0

# On your host, open browser to:
# http://localhost:6006
```

## Docker Run Options Explained

All run scripts use these optimized Docker options:

**GPU Scripts (`run.sh`, `run_training.sh`, `run_distributed.sh`):**
- `--gpus`: GPU device selection
- `--ipc=host`: Shared memory for faster data loading
- `--shm-size=16g`: 16GB shared memory for PyTorch DataLoader workers
- `--ulimit memlock=-1`: Unlimited locked memory
- `--ulimit stack=67108864`: Increased stack size for PyTorch
- `-v $(pwd):/workspace`: Mount entire workspace
- `-v <fast>:/storage/fast`: Mount fast storage (SSD/NVMe)
- `-v <slow>:/storage/slow`: Mount slow storage (HDD/Network)
- `-v <temp>:/storage/temp`: Mount temp storage (Scratch)
- `--rm`: Auto-remove container on exit (except for interactive mode)

**CPU Script (`run_cpu.sh`):**
- `--cpus`: All available CPU cores
- `--ipc=host`: Shared memory for data processing
- `--shm-size=16g`: 16GB shared memory for data loaders
- `-v $(pwd):/workspace`: Mount entire workspace
- `-v <fast>:/storage/fast`: Mount fast storage
- `-v <slow>:/storage/slow`: Mount slow storage
- `-v <temp>:/storage/temp`: Mount temp storage
- `--rm`: Auto-remove container on exit (except for interactive mode)
- No GPU access (for data preparation tasks)

## Advanced Usage

### Custom Docker Commands

If you need more control, you can run Docker commands directly:

```bash
# Run with custom environment variables
docker run -it --rm --gpus all \
    -v $(pwd):/workspace \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    -e WANDB_API_KEY=your_key \
    ts-dia-training:latest

# Run with additional volume mounts
docker run -it --rm --gpus all \
    -v $(pwd):/workspace \
    -v /path/to/external/data:/data \
    ts-dia-training:latest

# Run with specific Accelerate config
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    ts-dia-training:latest \
    accelerate launch --config_file accelerate_config.yaml train.py --config configs/train/training_example.yml
```

### Attach to Running Containers

If you started a training container and want to attach to it or run additional commands:

#### List Running Containers
```bash
# Show all running containers
docker ps

# Example output:
# CONTAINER ID   IMAGE                    NAMES
# abc123def456   ts-dia-training:latest   ts-dia-training-20251020-143022
```

#### Attach to Interactive Shell
```bash
# Attach to a running container with a new shell
docker exec -it <container_name> /bin/bash

# Example with specific container name
docker exec -it ts-dia-training-20251020-143022 /bin/bash

# Now you're inside the container - run any command
python train.py --config configs/train/training_example.yml
tensorboard --logdir /storage/fast/checkpoints
nvidia-smi
```

#### Run Commands Without Attaching
```bash
# Monitor GPU usage
docker exec -it <container_name> nvidia-smi

# Check training logs
docker exec -it <container_name> tail -f /workspace/checkpoints/*/logs/*.log

# Run a Python script
docker exec -it <container_name> python evaluate.py --config configs/eval.yml

# Check disk usage
docker exec -it <container_name> df -h

# List checkpoints
docker exec -it <container_name> ls -lh /storage/fast/checkpoints/
```

#### Multiple Shells in Same Container
```bash
# Terminal 1: Start training
./docker/run_training.sh configs/train/full_experiment.yml ts-dia-training my-exp

# Terminal 2: Attach to monitor
docker exec -it my-exp /bin/bash
# Inside: watch -n 1 nvidia-smi

# Terminal 3: Attach to check logs
docker exec -it my-exp /bin/bash
# Inside: tail -f /workspace/checkpoints/*/logs/*.log
```

### Build with Custom Options

Customize the build process:

```bash
# Build with specific Dockerfile
docker build -t my-image -f docker/dockerfile.nvidia .

# Build with build arguments
docker build -t my-image \
    --build-arg PYTORCH_VERSION=24.10 \
    -f docker/dockerfile.nvidia .

# Build without cache
docker build --no-cache -t my-image -f docker/dockerfile.nvidia .
```

## Troubleshooting

### Permission Denied on Scripts

```bash
# If you get "Permission denied" when running scripts
chmod +x docker/*.sh

# Or use the setup script
chmod +x docker/setup_permissions.sh
./docker/setup_permissions.sh
```

### GPU Not Detected

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# If fails, reinstall NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Permission Issues

```bash
# If you encounter permission issues with checkpoints/logs
chmod -R 777 checkpoints cache logs
```

### Out of Memory Errors

```bash
# Reduce batch size in your config
# Or use fewer GPUs
./docker/run_training.sh configs/train/training_example.yml ts-dia-training exp 0

# Monitor GPU memory
watch -n 1 nvidia-smi
```

### Container Name Conflicts

```bash
# If container name already exists
docker rm <container_name>

# Or use auto-generated names (don't specify container name)
./docker/run.sh ts-dia-training
```

## Environment Variables

You can pass environment variables to control training:

```bash
# Set CUDA devices
docker run -it --rm --gpus all \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -v $(pwd):/workspace \
    ts-dia-training:latest

# Set WandB API key
docker run -it --rm --gpus all \
    -e WANDB_API_KEY=your_key_here \
    -v $(pwd):/workspace \
    ts-dia-training:latest

# Set HuggingFace cache
docker run -it --rm --gpus all \
    -e HF_HOME=/workspace/cache/huggingface \
    -v $(pwd):/workspace \
    ts-dia-training:latest
```

## Best Practices

1. **Always build from project root:** Run scripts from `/workspaces/TS-DIA`
2. **Use named containers:** Easier to manage and attach to
   ```bash
   ./docker/run_training.sh configs/train/training_example.yml ts-dia-training my-experiment-v1 0,1
   # Later: docker exec -it my-experiment-v1 /bin/bash
   ```
3. **Monitor GPU usage:** Use `nvidia-smi` to track utilization
   ```bash
   docker exec -it <container_name> watch -n 1 nvidia-smi
   ```
4. **Check logs regularly:** Logs are saved in `checkpoints/*/logs/`
   ```bash
   docker exec -it <container_name> tail -f /workspace/checkpoints/*/logs/*.log
   ```
5. **Use distributed training:** For multi-GPU setups, use `run_distributed.sh`
6. **Backup checkpoints:** Checkpoints are saved to host, but backup important ones
7. **Attach for debugging:** Use `docker exec -it` to attach and run commands manually

## Container Management

```bash
# List all containers (running and stopped)
docker ps -a

# Stop a running container
docker stop <container_name>

# Remove a container
docker rm <container_name>

# View container logs
docker logs <container_name>

# List Docker images
docker images

# Remove unused images
docker image prune
```

## Support

For issues or questions:
1. Check the main project README
2. Review training configuration documentation
3. Check Accelerate documentation for distributed training issues
4. Verify GPU and Docker setup

## License

Licensed under Apache 2.0 license.

