export CUDA_VISIBLE_DEVICES="5"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model=llama-3-1-8b-instruct
model=mocorelora
experts=8

python train.py @configs/${base_model}_${model}_commonsense15k_train_exp${experts}.config

python test_math.py @configs/${base_model}_${model}_commonsense15k_test_exp${experts}.config

python evaluate_commonsense.py --predict_file /data/workspace/projects/moe/AdaMoLE/outputs/${base_model}-${model}-exp${experts}-commonsense-15k/predictions/boolq_responses.jsonl
