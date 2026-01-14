export CUDA_VISIBLE_DEVICES="2"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model=qwen3-14b
model=mocorelora
experts=4
seed=seed1225

python train.py @configs/${base_model}_${model}_math14k_train_exp${experts}_${seed}.config

python test_math.py @configs/${base_model}_${model}_math14k_test_exp${experts}_${seed}.config

python evaluate_math.py --predict_file /data/workspace/projects/moe/AdaMoLE/outputs/${base_model}-${model}-exp${experts}-${seed}-math-14k/predictions/addsub_responses.jsonl
