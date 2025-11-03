
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


[ ] Validate and Create descriptions for all the datasets. 

[ ] Write the Datasets sections in the paper. 

[ ] prep data simiulation useing mini_librispeech
    [x] Setup kaldi continer 
    [x] clone mini_librispeech recipe
    [x] prepare datasets with varing overlap threshold 
        [x] train 20k h 
        [x] 1 - 5 speakers N(1, 4)
            [x] overlap threshold modification 
            [x] atleast 100 h val for each combo
    [ ] ensure lhoste compatability 

[ ] setup training experiments 
    [ ] softmax attention
        [ ] validate all arguments for the training script are passed through
            [ ] model config checklist
                [ ] Top-level
                    [x] model_type: encoder_decoder  # Options: encoder, decoder, encoder_decoder
                    [x] name: comprehensive_example_model
                [ ] Global Configuration (shared)
                    [x] global_config.dropout: 0.1
                    [ ] global_config.batch_size: 128
                    [ ] global_config.d_ff: 4
                    [ ] global_config.device: cuda
                [ ] Encoder Configuration
                    [ ] encoder.d_model: 2048
                    [ ] encoder.num_layers: 4
                    [ ] encoder.num_heads: 8
                    [ ] encoder.attention_type: softmax / linear / causal_linear
                    [ ] encoder.nb_features: null or integer
                    [ ] encoder.activation: REGLU / GELU / RELU / SILU / GEGLU / SWIGLU
                    [ ] encoder.use_rezero: true / false
                    [ ] encoder.use_scalenorm: true / false
                    [ ] encoder.feature_redraw_interval: null or integer
                    [ ] encoder.auto_check_redraw: true / false

                [ ] Decoder Configuration
                    [ ] decoder.d_model: 2048
                    [ ] decoder.num_layers: 4
                    [ ] decoder.num_heads: 8
                    [ ] decoder.attention_type: softmax / linear / causal_linear
                    [ ] decoder.nb_features: null or integer
                    [ ] decoder.activation: REGLU / GELU / RELU / SILU / GEGLU / SWIGLU
                    [ ] decoder.use_cross_attention: true / false
                    [ ] decoder.use_rezero: true / false
                    [ ] decoder.use_scalenorm: true / false
                    [ ] decoder.feature_redraw_interval: null or integer
                    [ ] decoder.auto_check_redraw: true / false
                    [ ] decoder.num_classes: integer or null (classification head)

                [ ] Training: Basic
                    [ ] training.epochs: 100
                    [ ] training.batch_size: 128
                    [ ] training.random_seed: 42
                    [ ] training.max_steps: null or integer

                [ ] Optimizer
                    [ ] training.optimizer.type: adamw / adam / sgd / adagrad / rmsprop / adadelta
                    [ ] training.optimizer.lr: 0.0001
                    [ ] training.optimizer.weight_decay: 0.01
                    [ ] training.optimizer.betas: [0.9, 0.999]
                    [ ] training.optimizer.epsilon: 1e-8
                    [ ] training.optimizer.amsgrad: true / false
                    [ ] training.optimizer.momentum: null or float
                    [ ] training.optimizer.nesterov: true / false

                [ ] Learning Rate Scheduler
                    [ ] training.scheduler.type: cosine / linear / exponential / step / plateau / constant
                    [ ] training.scheduler.min_lr: 1e-6
                    [ ] training.scheduler.max_lr: null or float
                    [ ] training.scheduler.warmup_steps: integer
                    [ ] training.scheduler.decay_steps: integer
                    [ ] training.scheduler.num_cycles: integer (for cosine restarts)
                    [ ] training.scheduler.step_size: null or integer (StepLR)
                    [ ] training.scheduler.gamma: float (step decay)
                    [ ] training.scheduler.patience: null or integer (ReduceLROnPlateau)
                    [ ] training.scheduler.mode: min / max (ReduceLROnPlateau)

                [ ] Gradient Management
                    [ ] training.gradient_clipping: null or float
                    [ ] training.gradient_accumulation_steps: integer

                [ ] Mixed Precision / AMP
                    [ ] training.mixed_precision: true / false
                    [ ] training.amp_loss_scale: null or integer (dynamic if null)

                [ ] Performer / Linear-Attention Settings
                    [ ] training.feature_redraw_interval: null or integer
                    [ ] training.fixed_projection: true / false
                    [ ] encoder/decoder nb_features (if using linear attention)
                    [ ] checkpoint.snapshot_features: true / false (whether to persist random features)

                [ ] Loss Configuration
                    [ ] training.loss.main: cross_entropy / mse / mae / bce / bce_with_logits
                    [ ] training.loss.label_smoothing: float (0.0 = off)
                    [ ] training.loss.reduction: mean / sum / none
                    [ ] training.loss.auxiliary.norm_reg: null or float
                    [ ] training.loss.auxiliary.contrastive: null or float
                    [ ] training.loss.focal_alpha: null or float
                    [ ] training.loss.focal_gamma: null or float

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
                    [ ] training.checkpoint.save_dir: path
                    [ ] training.checkpoint.interval: steps
                    [ ] training.checkpoint.save_total_limit: integer
                    [ ] training.checkpoint.resume: null or checkpoint path
                    [ ] training.checkpoint.snapshot_optimizer: true / false
                    [ ] training.checkpoint.snapshot_scheduler: true / false
                    [ ] training.checkpoint.snapshot_features: true / false
                    [ ] training.checkpoint.save_best_only: true / false
                    [ ] training.checkpoint.monitor_metric: metric name

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

 

