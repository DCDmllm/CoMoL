export CUDA_VISIBLE_DEVICES="0"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model_path=Qwen/Qwen3-8B
base_model=qwen3-8b

output_path=outputs/${base_model}

python test_code10_basemodel.py \
    --model_path=${base_model_path} \
    --max_new_tokens=400 \
    --batch_size=16 

python evaluate_code.py --predict_file ${output_path}/predictions/humaneval_responses.jsonl
