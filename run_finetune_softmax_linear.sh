#!/bin/bash

# To run this script in the background and keep it running after closing the terminal:
# nohup ./run_finetune_softmax_linear.sh > logs/softmax_linear_model.log 2>&1 &

mkdir -p logs

# echo "Starting training for configs/FINETUNE/softmax_linear_model/softmax_linear_model_aishell4.yaml..."
# accelerate launch --config_file configs/accelerate/accelerate-8gpus.yaml train.py --config configs/FINETUNE/softmax_linear_model/softmax_linear_model_aishell4.yaml
# if [ $? -ne 0 ]; then
#     echo "Training failed for configs/FINETUNE/softmax_linear_model/softmax_linear_model_aishell4.yaml. Continuing to next job..."
# fi
# echo "--------------------------------------------------"

echo "Starting training for configs/FINETUNE/softmax_linear_model/softmax_linear_model_ami.yaml..."
accelerate launch --config_file configs/accelerate/accelerate-8gpus.yaml train.py --config configs/FINETUNE/softmax_linear_model/softmax_linear_model_ami.yaml
if [ $? -ne 0 ]; then
    echo "Training failed for configs/FINETUNE/softmax_linear_model/softmax_linear_model_ami.yaml. Continuing to next job..."
fi
echo "--------------------------------------------------"

echo "Starting training for configs/FINETUNE/softmax_linear_model/softmax_linear_model_ava_avd.yaml..."
accelerate launch --config_file configs/accelerate/accelerate-8gpus.yaml train.py --config configs/FINETUNE/softmax_linear_model/softmax_linear_model_ava_avd.yaml
if [ $? -ne 0 ]; then
    echo "Training failed for configs/FINETUNE/softmax_linear_model/softmax_linear_model_ava_avd.yaml. Continuing to next job..."
fi
echo "--------------------------------------------------"

echo "Starting training for configs/FINETUNE/softmax_linear_model/softmax_linear_model_combined.yaml..."
accelerate launch --config_file configs/accelerate/accelerate-8gpus.yaml train.py --config configs/FINETUNE/softmax_linear_model/softmax_linear_model_combined.yaml
if [ $? -ne 0 ]; then
    echo "Training failed for configs/FINETUNE/softmax_linear_model/softmax_linear_model_combined.yaml. Continuing to next job..."
fi
echo "--------------------------------------------------"

echo "Starting training for configs/FINETUNE/softmax_linear_model/softmax_linear_model_icsi.yaml..."
accelerate launch --config_file configs/accelerate/accelerate-8gpus.yaml train.py --config configs/FINETUNE/softmax_linear_model/softmax_linear_model_icsi.yaml
if [ $? -ne 0 ]; then
    echo "Training failed for configs/FINETUNE/softmax_linear_model/softmax_linear_model_icsi.yaml. Continuing to next job..."
fi
echo "--------------------------------------------------"

echo "Starting training for configs/FINETUNE/softmax_linear_model/softmax_linear_model_mswild.yaml..."
accelerate launch --config_file configs/accelerate/accelerate-8gpus.yaml train.py --config configs/FINETUNE/softmax_linear_model/softmax_linear_model_mswild.yaml
if [ $? -ne 0 ]; then
    echo "Training failed for configs/FINETUNE/softmax_linear_model/softmax_linear_model_mswild.yaml. Continuing to next job..."
fi
echo "--------------------------------------------------"

