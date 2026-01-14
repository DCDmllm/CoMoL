export CUDA_VISIBLE_DEVICES="3"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model=qwen3-14b
model=molora

python train.py @configs/${base_model}_${model}hydra_math14k_train.config

python test_math.py @configs/${base_model}_${model}hydra_math14k_test.config

python evaluate_math.py --predict_file /data/workspace/projects/moe/AdaMoLE/outputs/${base_model}-${model}-hydraTrue-exp4-math-14k/predictions/addsub_responses.jsonl
