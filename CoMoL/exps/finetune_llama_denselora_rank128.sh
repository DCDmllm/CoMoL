export CUDA_VISIBLE_DEVICES="3"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model=llama-3-1-8b-instruct
model=denselora
rank=rank128

python train.py @configs/${base_model}_${model}_math14k_train_${rank}.config

python test_math.py @configs/${base_model}_${model}_math14k_test_${rank}.config

python evaluate_math.py --predict_file /data/workspace/projects/moe/AdaMoLE/outputs/${base_model}-${model}-${rank}-math-14k/predictions/addsub_responses.jsonl
