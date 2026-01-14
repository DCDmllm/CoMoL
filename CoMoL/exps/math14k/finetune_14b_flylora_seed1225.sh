export CUDA_VISIBLE_DEVICES="2"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model=qwen3-14b
model=flylora
rank=rank32
seed=seed1225

python train.py @configs/math14k/${base_model}_${model}_train_${seed}.config

python test_math.py @configs/math14k/${base_model}_${model}_test_${seed}.config

python evaluate_math.py --predict_file outputs/${base_model}-${model}-${rank}-${seed}-math-14k/predictions/addsub_responses.jsonl
