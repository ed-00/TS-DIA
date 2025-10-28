# Docker Setup Summary for TS-DIA

## Quick Reference

### Initial Setup (One-time)

```bash
# 0. Make scripts executable (first time only)
chmod +x docker/*.sh
# OR: chmod +x docker/setup_permissions.sh && ./docker/setup_permissions.sh

# 1. Build the Docker image
./docker/build.sh

# 2. (Optional) Configure storage volumes
cp docker/volumes.env.example docker/volumes.env
# Edit docker/volumes.env with your storage paths
```

### Running Training

```bash
# Interactive session (all 8 GPUs)
./docker/run.sh

# Direct training (all 8 GPUs)
./docker/run_training.sh configs/train/training_example.yml

# Distributed training (4 out of 8 GPUs)
./docker/run_distributed.sh configs/train/training_example.yml 4

# Distributed training (all 8 GPUs)
./docker/run_distributed.sh configs/train/training_example.yml 8

# CPU-only for data prep
./docker/run_cpu.sh
```

## Files Created

### Core Files
- **`dockerfile.nvidia`** - Main Dockerfile using NVIDIA PyTorch base
- **`build.sh`** - Build script with image reuse
- **`run.sh`** - Interactive GPU container
- **`run_training.sh`** - Direct training execution
- **`run_distributed.sh`** - Multi-GPU distributed training
- **`run_cpu.sh`** - CPU-only for data preparation
- **`README.md`** - Complete documentation

### Configuration Files
- **`volume_config.sh`** - Volume configuration loader
- **`volumes.env.example`** - Example storage configuration
- **`volumes.env`** - Your custom storage paths (gitignored)

### Accelerate Configs
- **`../configs/accelerate_2gpu_test.yaml`** - 2 GPU testing configuration
- **`../configs/accelerate_ddp.yaml`** - 8 GPU DDP configuration
- **`../configs/accelerate_fsdp.yaml`** - 8 GPU FSDP configuration

## Volume Mapping

### Inside Container

| Container Path | Purpose | Host Path (Configurable) |
|---------------|---------|--------------------------|
| `/workspace` | Project root | Current directory |
| `/storage/fast` | Fast storage (SSD/NVMe) | Configured or `./storage/fast` |
| `/storage/slow` | Slow storage (HDD/Network) | Configured or `./storage/slow` |
| `/storage/temp` | Temp/scratch space | Configured or `./storage/temp` |

## Script Parameters

### `build.sh [IMAGE_NAME] [--force]`
- **IMAGE_NAME**: Docker image name (default: `ts-dia-training`)
- **--force**: Force rebuild even if exists

### `run.sh [IMAGE_NAME] [CONTAINER_NAME] [GPU_IDS]`
- **IMAGE_NAME**: Docker image name
- **CONTAINER_NAME**: Container name
- **GPU_IDS**: `all` (all 8 GPUs), `0`, `0,1`, `0,1,2,3`, `0,1,2,3,4,5,6,7`, etc.

### `run_training.sh [CONFIG_FILE] [IMAGE_NAME] [CONTAINER_NAME] [GPU_IDS]`
- **CONFIG_FILE**: Training config YAML path
- **IMAGE_NAME**: Docker image name
- **CONTAINER_NAME**: Container name
- **GPU_IDS**: GPU selection

### `run_distributed.sh [CONFIG_FILE] [NUM_GPUS] [IMAGE_NAME] [CONTAINER_NAME] [ACCELERATE_CONFIG]`
- **CONFIG_FILE**: Training config YAML path
- **NUM_GPUS**: Number of GPUs - e.g., 4, 8 (auto-detected by default, will use all 8)
- **IMAGE_NAME**: Docker image name
- **CONTAINER_NAME**: Container name
- **ACCELERATE_CONFIG**: Path to Accelerate config file (optional)

### `run_cpu.sh [IMAGE_NAME] [CONTAINER_NAME] [COMMAND]`
- **IMAGE_NAME**: Docker image name
- **CONTAINER_NAME**: Container name
- **COMMAND**: Command to run (default: `/bin/bash` for interactive)

## Common Usage Patterns

### Development Workflow
```bash
# Start interactive session
./docker/run.sh

# Inside container
python train.py --config configs/train/training_example.yml
```

### Testing and Production Training
```bash
# Single GPU (GPU 0) for quick testing
./docker/run_training.sh configs/train/training_example.yml ts-dia-training exp1 0

# 2-GPU testing before full training
./docker/run_distributed.sh configs/train/training_example.yml 2 ts-dia-training test configs/accelerate_2gpu_test.yaml

# Multi-GPU with DDP (4 out of 8 GPUs)
./docker/run_distributed.sh configs/train/full_experiment.yml 4

# All 8 GPUs with DDP (production)
./docker/run_distributed.sh configs/train/full_experiment.yml 8

# Multi-GPU with FSDP (large models, all 8 GPUs)
./docker/run_distributed.sh configs/train/encoder_model.yml 8 ts-dia-training exp configs/accelerate_fsdp.yaml
```

### Data Preparation
```bash
# Interactive CPU session
./docker/run_cpu.sh

# Direct command execution
./docker/run_cpu.sh ts-dia-training prep "python data_manager/data_manager.py --dataset voxconverse"
```

## Storage Configuration Examples

### Single Machine with NVMe + HDD
```bash
# docker/volumes.env
FAST_STORAGE=/mnt/nvme/ts-dia
SLOW_STORAGE=/mnt/hdd/datasets
TEMP_STORAGE=/tmp/ts-dia
```

### Cloud Instance with Multiple Disks
```bash
# docker/volumes.env
FAST_STORAGE=/mnt/disks/ssd/ts-dia
SLOW_STORAGE=/mnt/disks/hdd/datasets
TEMP_STORAGE=/mnt/disks/scratch/ts-dia
```

### Local Development (Defaults)
```bash
# Don't create docker/volumes.env
# Automatically uses:
# - ./storage/fast
# - ./storage/slow
# - ./storage/temp
```

## Using Storage in Your Code

### Python Example
```python
# In your training script
import os

FAST_STORAGE = "/storage/fast"
SLOW_STORAGE = "/storage/slow"
TEMP_STORAGE = "/storage/temp"

# Save checkpoints to fast storage
checkpoint_dir = os.path.join(FAST_STORAGE, "checkpoints", "experiment1")

# Load datasets from slow storage
dataset_path = os.path.join(SLOW_STORAGE, "voxconverse")

# Cache to temp storage
cache_dir = os.path.join(TEMP_STORAGE, "cache")
```

### Config YAML Example
```yaml
checkpoint:
  save_dir: /storage/fast/checkpoints/my-experiment

dataset:
  data_path: /storage/slow/datasets/voxconverse
  
features:
  cache_dir: /storage/temp/feature_cache
```

## Docker Options Used

All scripts include these optimized settings:
- `--gpus`: GPU device selection (GPU scripts only)
- `--ipc=host`: Shared memory for faster data loading
- `--shm-size=16g`: 16GB shared memory for PyTorch DataLoader
- `--ulimit memlock=-1`: Unlimited locked memory
- `--ulimit stack=67108864`: Increased stack size
- Volume mounts for workspace and storage
- Port forwarding: 6006 (TensorBoard), 8888 (Jupyter)

## Troubleshooting

### Image Rebuild Needed
```bash
# After updating requirements.txt
./docker/build.sh --force
```

### Check Volume Mounts
```bash
# Inside container
ls -la /storage/fast
ls -la /storage/slow
ls -la /storage/temp
```

### GPU Not Detected
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Permission Issues
```bash
# Fix permissions on host
chmod -R 777 storage/
```

## Attaching to Running Containers

### List Running Containers
```bash
docker ps
```

### Attach to Running Container
```bash
# Get container name from docker ps
docker exec -it <container_name> /bin/bash

# Example
docker exec -it my-experiment /bin/bash

# Now run commands inside
python train.py --config configs/train/training_example.yml
nvidia-smi
tensorboard --logdir /storage/fast/checkpoints
```

### Run Commands Without Attaching
```bash
# Monitor GPUs
docker exec -it <container_name> nvidia-smi

# Check logs
docker exec -it <container_name> tail -f /workspace/checkpoints/*/logs/*.log

# Run evaluation
docker exec -it <container_name> python evaluate.py --config configs/eval.yml

# List checkpoints
docker exec -it <container_name> ls -lh /storage/fast/checkpoints/
```

### Multiple Terminals on Same Container
```bash
# Terminal 1: Start training with a named container
./docker/run_training.sh configs/train/full_experiment.yml ts-dia-training my-exp 0,1

# Terminal 2: Monitor GPUs
docker exec -it my-exp watch -n 1 nvidia-smi

# Terminal 3: Monitor logs
docker exec -it my-exp tail -f /workspace/checkpoints/*/logs/*.log

# Terminal 4: Run evaluation during training
docker exec -it my-exp python evaluate.py --checkpoint /storage/fast/checkpoints/latest
```

## Best Practices

1. **Use storage volumes appropriately:**
   - Fast: Active checkpoints, models being trained
   - Slow: Large datasets, archives, backups
   - Temp: Caching, temporary processing files

2. **Use named containers for easy access:**
   ```bash
   ./docker/run_training.sh configs/train/training_example.yml ts-dia-training my-experiment 0,1
   docker exec -it my-experiment /bin/bash
   ```

3. **Attach for monitoring:**
   ```bash
   docker exec -it <container_name> watch -n 1 nvidia-smi
   ```

4. **Clean up temp storage regularly:**
   ```bash
   rm -rf storage/temp/*
   ```

5. **Back up important checkpoints:**
   ```bash
   cp -r storage/fast/checkpoints/best-model /backup/location/
   ```

6. **Monitor disk usage:**
   ```bash
   df -h
   du -sh storage/*
   ```

## Additional Resources

- Full documentation: `docker/README.md`
- Training configs: `configs/train/`
- Accelerate configs: `configs/accelerate_*.yaml`
- Main training script: `train.py`

