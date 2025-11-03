
# Data prep plan 
Week 43, the transformer model is implementd, the data need to be preped to start training on the diarization data. 

[x]  Decide on a preprocessing standard for all the datasets 
    [x] Choose sample rate (8khz becasue many other studies use that) 
    [x] Choose window and overlap, 25ms window 10ms overlap (AED-EEND uses that) 
    [x] Choose feature dimentions, 345 feature dimentitions ie 21 feature bins
    [x] Choose locaiton for the data (/sbt-fast/dia-df/TS-DIA-2025-10/TS-DIA/data) 

[ ] Create a diarization labels converstion function 
    [ ] For each speaker in a sample, repeat the fueatures N number of speaker time
    [ ] Convert diarization lables from R^(SxT) to (C x T)
    [ ] Store the altered lables while keeping the audio files untouched 

[x] Run the prep function or all the datasets 
    <!-- [ ] LibriheavyMix 
        [x] yaml
        [ ] download
        [ ] validate -->
    [x] Ami 
        [x] yaml
        [x] download
        [x] validate
    [x] ICSI 
        [x] yaml
        [x] download
        [x] validate
    [x] AISHELL-4
        [x] yaml
        [x] download
        [x] validate
    [x] VoxConverse
        [x] yaml
        [x] download
        [x] validate
    [x] Ava-Avd
        [x] yaml
        [x] download
        [x] validate 
    [x] mswild 
        [x] yaml
        [x] download
        [x] validate 

    [ ] DIHARD III
        [ ] yaml
        [ ] download
        [ ] validate 
    [ ] CHiME-6
        [ ] yaml
        [ ] download
        [ ] validate 
    [ ] LibriCSS
        [ ] yaml
        [ ] download
        [ ] validate 
    [ ] TED-LIUM v2/v3
        [ ] yaml
        [ ] download
        [ ] validate 
    [ ] VoxCeleb1 & 2
        [ ] yaml
        [ ] download
        [ ] validate 
    <!-- [ ] Ego-4d
        [x] Apply for access again
        [x] yaml
        [ ] download
        [ ] validate -->


[x] Validate and Create descriptions for all the datasets. 

[ ] Write the Datasets sections in the paper. 

[x] prep data simiulation using mini_librispeech
    [x] Setup kaldi continer 
    [x] clone mini_librispeech recipe
    [x] prepare datasets with varing overlap threshold 
        [x] train 20k h 
        [x] 1 - 5 speakers N(1, 4)
            [x] overlap threshold modification 
            [x] atleast 100 h val for each combo
    [x] ensure lhoste compatability 

[x] setup training experiments 
    [x] softmax attention
        [x] validate all arguments for the training script are passed through
            [x] model config checklist
                [x] Top-level
                    [x] model_type: encoder_decoder  # Options: encoder, decoder, encoder_decoder
                    [x] name: comprehensive_example_model
                [x] Global Configuration (shared)
                    [x] global_config.dropout: 0.1
                    [x] global_config.batch_size: 128
                    [x] global_config.d_ff: 4
                    [x] global_config.device: cuda
                [x] Encoder Configuration
                    [x] encoder.d_model: 2048
                    [x] encoder.num_layers: 4
                    [x] encoder.num_heads: 8
                    [x] encoder.attention_type: softmax / linear / causal_linear
                    [x] encoder.nb_features: null or integer
                    [x] encoder.activation: REGLU / GELU / RELU / SILU / GEGLU / SWIGLU
                    [x] encoder.use_rezero: true / false
                    [x] encoder.use_scalenorm: true / false
                    [x] encoder.feature_redraw_interval: null or integer
                    [x] encoder.auto_check_redraw: true / false
                [x] Decoder Configuration
                    [x] decoder.d_model: 2048
                    [x] decoder.num_layers: 4
                    [x] decoder.num_heads: 8
                    [x] decoder.attention_type: softmax / linear / causal_linear
                    [x] decoder.nb_features: null or integer
                    [x] decoder.activation: REGLU / GELU / RELU / SILU / GEGLU / SWIGLU
                    [x] decoder.use_cross_attention: true / false
                    [x] decoder.use_rezero: true / false
                    [x] decoder.use_scalenorm: true / false
                    [x] decoder.feature_redraw_interval: null or integer
                    [x] decoder.auto_check_redraw: true / false
                    [x] decoder.num_classes: integer or null (classification head)
                [x] Training: Basic
                    [x] training.epochs: 100
                    [x] training.batch_size: 128
                    [x] training.random_seed: 42
                    [x] training.max_steps: null or integer
                [x] Optimizer
                    [x] training.optimizer.type: adamw / adam / sgd / adagrad / rmsprop / adadelta
                    [x] training.optimizer.lr: 0.0001
                    [x] training.optimizer.weight_decay: 0.01
                    [x] training.optimizer.betas: [0.9, 0.999]
                    [x] training.optimizer.epsilon: 1e-8
                    [x] training.optimizer.amsgrad: true / false
                    [x] training.optimizer.momentum: null or float
                    [x] training.optimizer.nesterov: true / false
                [x] Learning Rate Scheduler
                    [x] training.scheduler.type: cosine / linear / exponential / step / plateau / constant
                    [x] training.scheduler.min_lr: 1e-6
                    [x] training.scheduler.max_lr: null or float
                    [x] training.scheduler.warmup_steps: integer
                    [x] training.scheduler.decay_steps: integer
                    [x] training.scheduler.num_cycles: integer (for cosine restarts)
                    [x] training.scheduler.step_size: null or integer (StepLR)
                    [x] training.scheduler.gamma: float (step decay)
                    [x] training.scheduler.patience: null or integer (ReduceLROnPlateau)
                    [x] training.scheduler.mode: min / max (ReduceLROnPlateau)
                [x] Gradient Management
                    [x] training.gradient_clipping: null or float
                    [x] training.gradient_accumulation_steps: integer
                [x] Mixed Precision / AMP
                    [x] training.mixed_precision: true / false
                    [x] training.amp_loss_scale: null or integer (dynamic if null)
                [x] Performer / Linear-Attention Settings
                    [x] training.feature_redraw_interval: null or integer
                    [x] training.fixed_projection: true / false
                    [x] encoder/decoder nb_features (if using linear attention)
                    [x] checkpoint.snapshot_features: true / false (whether to persist random features)
                [x] Loss Configuration
                    [x] training.loss.main: cross_entropy / mse / mae / bce / bce_with_logits
                    [x] training.loss.label_smoothing: float (0.0 = off)
                    [x] training.loss.reduction: mean / sum / none
                    [x] training.loss.auxiliary.norm_reg: null or float
                    [x] training.loss.auxiliary.contrastive: null or float
                    [x] training.loss.focal_alpha: null or float
                    [x] training.loss.focal_gamma: null or float
                [ ] Validation
                    [ ] training.validation.interval: integer steps
                    [ ] training.validation.batch_size: integer
                    [ ] training.validation.metric_for_best_model: e.g., val_loss
                    [ ] training.validation.greater_is_better: true / false
                [ ] Early Stopping
                    [ ] training.early_stopping.patience: integer epochs
                    [ ] training.early_stopping.metric: metric name
                    [ ] training.early_stopping.min_delta: float
                    [ ] training.early_stopping.mode: min / max
                    [ ] training.early_stopping.restore_best_weights: true / false
                [ ] Checkpointing
                    [x] training.checkpoint.save_dir: path
                    [x] training.checkpoint.interval: steps
                    [x] training.checkpoint.save_total_limit: integer
                    [x] training.checkpoint.resume: null or checkpoint path
                    [ ] training.checkpoint.snapshot_optimizer: true / false
                    [x] training.checkpoint.snapshot_scheduler: true / false
                    [x] training.checkpoint.snapshot_features: true / false
                    [ ] training.checkpoint.save_best_only: true / false
                    [x] training.checkpoint.monitor_metric: metric name
                [ ] Distributed Training
                    [ ] training.distributed.backend: nccl / gloo / mpi
                    [ ] training.distributed.world_size: integer
                    [ ] training.distributed.local_rank: integer
                    [ ] training.distributed.sync_gradient_barrier: true / false
                    [ ] training.distributed.find_unused_parameters: true / false
                    [ ] training.distributed.gradient_as_bucket_view: true / false
                    [ ] training.distributed.static_graph: true / false
                [ ] Logging
                    [ ] training.logging.interval: steps
                    [ ] training.logging.tensorboard: true / false
                    [ ] training.logging.wandb: true / false
                    [ ] training.logging.wandb_project: project name
                    [ ] training.logging.wandb_entity: entity/team
                    [ ] training.logging.log_model: true / false
                [ ] Performance / DataLoader
                    [ ] training.performance.num_workers: integer
                    [ ] training.performance.pin_memory: true / false
                    [ ] training.performance.batch_shim: true / false
                    [ ] training.performance.prefetch_factor: integer
                    [ ] training.performance.persistent_workers: true / false
                    [ ] training.performance.compile_model: true / false
                [ ] Callbacks
                    [ ] training.callbacks: list includes gradient_clipping, pruning, freeze_layers, dynamic_lr, etc.
                [ ] Profiling
                    [ ] training.profiling: true / false
                [ ] Evaluation Knobs (inference / diarization)
                    [ ] eval_knobs.batch_size: integer
                    [ ] eval_knobs.beam_width: integer
                    [ ] eval_knobs.sliding_window: integer
                    [ ] eval_knobs.label_type: binary / speaker_id / custom
                    [ ] eval_knobs.max_duration: null or float

                [ ] Hyperparameter Tuning
                    [ ] tuning.library: raytune / optuna / null
                    [ ] tuning.search_space: ensure keys and ranges are defined (lr, batch_size, dropout, etc.)
        [ ] Ensure the Ego-dataset is being used 
        [ ] Test run speed, utilization, and select optimal:
            [ ] batch size
            [ ] dataloader config
        [ ] validate learning rate scheduler is working properly
        [ ] validate hyper-parameters 
            [ ] learning rate
            [ ] weight decay
            [ ] gradient clipping
            [ ] lable smoothing

    [ ] linear attention 

 

