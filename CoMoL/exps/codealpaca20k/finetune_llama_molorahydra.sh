export CUDA_VISIBLE_DEVICES="0"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model=llama-3-1-8b-instruct
model=molorahydra
dataset=codealpaca20k

python train.py @configs/${dataset}/${base_model}_${model}_train.config

# python test_math.py @configs/${base_model}_${model}_math14k_test.config

# python evaluate_math.py --predict_file /data/workspace/projects/moe/AdaMoLE/outputs/${base_model}-${model}-math-14k/predictions/addsub_responses.jsonl
