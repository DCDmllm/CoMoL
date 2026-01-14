export CUDA_VISIBLE_DEVICES="0"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model=llama-3-1-8b-instruct
model=mocorelora
experts=8

python train.py @configs/${base_model}_${model}_math14k_train_exp${experts}.config

python test_math.py @configs/${base_model}_${model}_math14k_test_exp${experts}.config

python evaluate_math.py --predict_file /data/workspace/projects/moe/AdaMoLE/outputs/${base_model}-${model}-exp${experts}-math-14k/predictions/addsub_responses.jsonl
