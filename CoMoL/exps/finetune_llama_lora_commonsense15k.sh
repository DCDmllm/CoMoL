export CUDA_VISIBLE_DEVICES="3"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model=llama-3-1-8b-instruct
model=lora

python train.py @configs/${base_model}_${model}_commonsense15k_train.config

python test_math.py @configs/${base_model}_${model}_commonsense15k_test.config

python evaluate_commonsense.py --predict_file /data/workspace/projects/moe/AdaMoLE/outputs/${base_model}-${model}-commonsense-15k/predictions/boolq_responses.jsonl
