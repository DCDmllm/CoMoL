export CUDA_VISIBLE_DEVICES="5"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model=qwen3-8b
model=flylora
rank=rank32

python train.py @configs/math14k/${base_model}_${model}_train.config

python test_math.py @configs/math14k/${base_model}_${model}_test.config

python evaluate_math.py --predict_file outputs/${base_model}-${model}-${rank}-math-14k/predictions/addsub_responses.jsonl
