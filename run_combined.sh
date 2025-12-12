echo "Starting training for configs/FINETUNE/softmax_pretraining_model/softmax_pretraining_model_combined.yaml..."
accelerate launch --config_file configs/accelerate/accelerate-8gpus.yaml train.py --config configs/FINETUNE/softmax_pretraining_model/softmax_pretraining_model_combined.yaml
if [ $? -ne 0 ]; then
    echo "Training failed for configs/FINETUNE/softmax_pretraining_model/softmax_pretraining_model_combined.yaml. Continuing to next job..."
fi
echo "--------------------------------------------------"

echo "Starting training for configs/FINETUNE/softmax_linear_model_correct_nbf/softmax_linear_model_correct_nbf_combined.yaml..."
accelerate launch --config_file configs/accelerate/accelerate-8gpus.yaml train.py --config configs/FINETUNE/softmax_linear_model_correct_nbf/softmax_linear_model_correct_nbf_combined.yaml
if [ $? -ne 0 ]; then
    echo "Training failed for configs/FINETUNE/softmax_linear_model_correct_nbf/softmax_linear_model_correct_nbf_combined.yaml. Continuing to next job..."
fi
echo "--------------------------------------------------"


echo "Starting training for configs/FINETUNE/softmax_linear_model/softmax_linear_model_combined.yaml..."
accelerate launch --config_file configs/accelerate/accelerate-8gpus.yaml train.py --config configs/FINETUNE/softmax_linear_model/softmax_linear_model_combined.yaml
if [ $? -ne 0 ]; then
    echo "Training failed for configs/FINETUNE/softmax_linear_model/softmax_linear_model_combined.yaml. Continuing to next job..."
fi
echo "--------------------------------------------------"