export CUDA_VISIBLE_DEVICES="3"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model=llama-3-1-8b-instruct
model=mocorelora-corerouter
dataset=codealpaca20k
exp=exp4

python train.py @configs/${dataset}/${base_model}_${model}_${exp}_train.config

python test_code10.py @configs/${dataset}/${base_model}_${model}_${exp}_test.config

python evaluate_code.py --predict_file outputs/${base_model}-${model}-${exp}-${dataset}/predictions/humaneval_responses.jsonl


python test_code10.py @configs/${dataset}/${base_model}_${model}_${exp}_test_4epoch.config
python test_code10.py @configs/${dataset}/${base_model}_${model}_${exp}_test_3epoch.config
python test_code10.py @configs/${dataset}/${base_model}_${model}_${exp}_test_2epoch.config
python test_code10.py @configs/${dataset}/${base_model}_${model}_${exp}_test_1epoch.config

python evaluate_code.py --predict_file outputs/${base_model}-${model}-${exp}-${dataset}/checkpoint-5008/predictions/humaneval_responses.jsonl
python evaluate_code.py --predict_file outputs/${base_model}-${model}-${exp}-${dataset}/checkpoint-3756/predictions/humaneval_responses.jsonl
python evaluate_code.py --predict_file outputs/${base_model}-${model}-${exp}-${dataset}/checkpoint-2504/predictions/humaneval_responses.jsonl
python evaluate_code.py --predict_file outputs/${base_model}-${model}-${exp}-${dataset}/checkpoint-1252/predictions/humaneval_responses.jsonl
