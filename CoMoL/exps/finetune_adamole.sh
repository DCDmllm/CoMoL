export CUDA_VISIBLE_DEVICES="4"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model=qwen3-8b
model=adamole

python train.py @configs/${base_model}_${model}_math14k_train.config

python test_math.py @configs/${base_model}_${model}_math14k_test.config

python evaluate_math.py --predict_file /data/workspace/projects/moe/AdaMoLE/outputs/${base_model}-${model}-the8-math-14k/predictions/addsub_responses.jsonl
