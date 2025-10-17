# Experiment Management System

## Overview

The TS-DIA training system includes comprehensive experiment management features to ensure all training runs are properly tracked, logged, and reproducible. All experiment artifacts are automatically organized in the checkpoint directory.

## Key Features

### 1. Checkpoint Directory Protection

**Prevents accidental overwriting of experiments**

- If a checkpoint directory already exists and you're not resuming, training will fail with a clear error message
- You must explicitly choose to:
  - Change the `checkpoint.save_dir` to a new path
  - Set `checkpoint.resume` to resume from the existing checkpoint
  - Manually delete the directory if intentional

**Example Error:**
```
======================================================================
ERROR: Checkpoint directory already exists!
======================================================================
Directory: ./checkpoints/my_experiment

To prevent accidental overwriting of existing experiments, training cannot proceed.

Please choose ONE of the following options:
  1. Change 'checkpoint.save_dir' in your config to a new path
  2. Set 'checkpoint.resume' to resume from this checkpoint
  3. Manually delete the directory if you want to overwrite it
======================================================================
```

### 2. Configuration File Preservation

**Every experiment saves its exact configuration**

- The YAML configuration file is automatically copied to `{checkpoint_dir}/config.yml`
- Ensures complete reproducibility
- Makes it easy to see exactly how an experiment was configured

### 3. Complete Console Logging

**All console output is saved to log files**

- New runs: `{checkpoint_dir}/logs/training_{timestamp}.log`
- Resumed runs: `{checkpoint_dir}/logs/training_resume_{N}_{timestamp}.log`
- Captures all print statements, progress bars, and validation outputs
- Logs are written to both console and file simultaneously

### 4. Full Config Sent to WandB

**WandB experiments include complete configuration**

- The entire YAML config (model, dataset, training) is sent to WandB
- Not just flattened training parameters
- Allows complete experiment tracking and comparison in WandB UI

### 5. WandB Run Information

**WandB run details are saved locally**

- `{checkpoint_dir}/wandb_info.json` contains:
  - Run ID
  - Run name
  - Run URL
  - Project name
  - Entity name
  - Timestamp

**Example:**
```json
{
  "run_id": "abc123xyz",
  "run_name": "experiment_name",
  "run_url": "https://wandb.ai/entity/project/runs/abc123xyz",
  "project": "ts-dia-training",
  "entity": "my-team",
  "timestamp": "2025-10-16T11:01:47"
}
```

### 6. Organized Directory Structure

**All experiment artifacts in one place**

```
checkpoints/my_experiment/
├── config.yml                          # Original configuration
├── wandb_info.json                     # WandB run information (if enabled)
├── logs/                               # Training logs
│   ├── training_20251016_110147.log    # Initial training log
│   └── training_resume_1_20251016_120534.log  # Resume log
├── checkpoints/                        # Accelerate's checkpoint storage
├── checkpoint-epoch0-step100/          # Named checkpoint
├── checkpoint-epoch1-step200/          # Named checkpoint
├── best_checkpoint -> checkpoint-epoch1-step200  # Symlink to best
├── hyperparameters.json                # Training hyperparameters
└── model_summary.txt                   # Model architecture
```

### 7. Enhanced Validation Logging

**Validation metrics properly logged to WandB/TensorBoard**

- Validation metrics are logged with proper prefixes (`val_loss`, `val_accuracy`, etc.)
- Confirmation message shows when metrics are logged
- Metrics aligned with training step for easy comparison

## Usage Examples

### Basic Training Run

```yaml
# config.yml
training:
  checkpoint:
    save_dir: ./checkpoints/experiment_001
    interval: 1000
  
  logging:
    interval: 50
    wandb: true
    wandb_project: my-project
```

```bash
# First run - creates directory and all artifacts
python train.py --config config.yml

# Second run - fails with error
python train.py --config config.yml
# ERROR: Checkpoint directory already exists!

# To start new experiment, change the path
# In config.yml, change to: save_dir: ./checkpoints/experiment_002
python train.py --config config.yml
```

### Resuming Training

```yaml
# config.yml
training:
  checkpoint:
    save_dir: ./checkpoints/experiment_001
    resume: ./checkpoints/experiment_001/checkpoint-epoch5-step1000
    interval: 1000
```

```bash
# Resume from checkpoint
python train.py --config config.yml

# Check logs
ls checkpoints/experiment_001/logs/
# training_20251016_110147.log
# training_resume_1_20251016_120534.log
```

### Inspecting Experiment Artifacts

```bash
# View configuration used for experiment
cat checkpoints/experiment_001/config.yml

# View WandB run information
cat checkpoints/experiment_001/wandb_info.json

# View training logs
tail -f checkpoints/experiment_001/logs/training_*.log

# List all checkpoints
ls -lh checkpoints/experiment_001/checkpoint-*

# View model architecture
cat checkpoints/experiment_001/model_summary.txt
```

## Best Practices

### 1. Naming Experiments

Use descriptive checkpoint directory names:

```yaml
checkpoint:
  save_dir: ./checkpoints/ava_8khz_linear_attention_v1
```

Good naming helps you:
- Identify experiments quickly
- Understand what was tested
- Organize related experiments

### 2. WandB Integration

Enable WandB for comprehensive tracking:

```yaml
logging:
  wandb: true
  wandb_project: ts-dia-research
  wandb_entity: my-team
  tensorboard: true  # Can use both simultaneously
```

### 3. Regular Checkpointing

Set appropriate checkpoint intervals:

```yaml
checkpoint:
  interval: 1000          # Save every 1000 steps
  save_total_limit: 5     # Keep only last 5 checkpoints
```

### 4. Validation Monitoring

Configure validation to run regularly:

```yaml
validation:
  interval: 1             # Validate every epoch
  batch_size: 32
  max_steps: 100          # Limit validation steps for speed
```

## Troubleshooting

### Checkpoint Directory Exists Error

**Problem:** Getting error about existing checkpoint directory

**Solution:**
1. Change `checkpoint.save_dir` to a new path
2. Add `checkpoint.resume` to resume from existing checkpoint
3. Delete the directory: `rm -rf checkpoints/my_experiment`

### Logs Not Appearing

**Problem:** Console output not being saved to log files

**Solution:**
- Ensure `checkpoint.save_dir` is configured
- Check that logs directory exists: `ls checkpoints/my_experiment/logs/`
- Verify you have write permissions

### WandB Info Not Saved

**Problem:** `wandb_info.json` not created

**Solution:**
- Ensure WandB is enabled: `logging.wandb: true`
- Verify checkpoint configuration is present
- Check WandB authentication: `wandb login`

### Validation Metrics Missing in WandB

**Problem:** Validation metrics don't appear in WandB UI

**Solution:**
- Verify logging is enabled: `logging.wandb: true`
- Check validation is running: Set `validation.interval: 1`
- Look for confirmation message: "✓ Validation metrics logged to trackers"
- Metrics should appear under "Charts" in WandB run page

## Implementation Details

### Modified Files

1. **`parse_args.py`**: Returns config file path alongside parsed configs
2. **`train.py`**: Passes config path to Trainer
3. **`training/trainer.py`**: 
   - Validates checkpoint directory
   - Copies config file
   - Sets up file logging
   - Saves WandB info
4. **`training/logging_utils.py`**:
   - Enhanced `init_trackers()` to send full config to WandB
   - Added `save_wandb_info()` function
   - Added `setup_file_logger()` function

### Error Classes

- **`CheckpointDirectoryExistsError`**: Raised when checkpoint directory exists without resume flag

### Functions

- **`_validate_checkpoint_directory()`**: Validates checkpoint directory doesn't exist
- **`_copy_config_file()`**: Copies configuration to checkpoint directory
- **`setup_file_logger()`**: Sets up dual console and file logging
- **`save_wandb_info()`**: Extracts and saves WandB run information

## Migration Guide

If you have existing code using the old trainer:

### Before
```python
from training import Trainer

trainer = Trainer(
    model=model,
    train_dataloader=train_dl,
    val_dataloader=val_dl,
    config=training_config,
)
```

### After
```python
from training import Trainer

# Now pass config_path for full experiment tracking
trainer = Trainer(
    model=model,
    train_dataloader=train_dl,
    val_dataloader=val_dl,
    config=training_config,
    config_path=config_path,  # New parameter
)
```

The config_path is automatically handled by `train.py` when using `unified_parser()`.

## Summary

The experiment management system ensures that:
- ✅ Experiments cannot be accidentally overwritten
- ✅ Every experiment's configuration is preserved
- ✅ All console output is logged to files
- ✅ WandB receives complete experiment configuration
- ✅ WandB run information is saved locally
- ✅ All artifacts are organized in checkpoint directory
- ✅ Validation metrics are properly tracked

This makes experiment management simple, reproducible, and hassle-free!

