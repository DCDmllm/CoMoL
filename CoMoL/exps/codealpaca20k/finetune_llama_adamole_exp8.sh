export CUDA_VISIBLE_DEVICES="1"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model_path=meta-llama/Llama-3.1-8B-Instruct
base_model=llama-3-1-8b-instruct
dataset_path=/data/workspace/projects/moe/datasets/codealpaca20k
dataset=codealpaca20k
model=adamole
rank=8
experts=8

python train.py \
    --model_path=${base_model_path} \
    --data_path=${dataset_path} \
    --peft_type=${model} \
    --lora_rank=${rank} \
    --target_modules \
    q_proj \
    k_proj \
    v_proj \
    o_proj \
    down_proj \
    --num_experts=${experts} \
    --threshold=0.125 \
    --max_length=500 \
    --batch_size=1 \
    --gradient_accumulation_steps=16 \
    --num_train_epochs=1 \
    --learning_rate=1e-4 \
    --lr_scheduler_type=constant_with_warmup \
    --warmup_steps=200 \
    --weight_decay=0.0 \
    --aux_loss_coeff=1e-3

output_path=outputs/${base_model}-${model}-rank${rank}-exp${experts}-${dataset}

python test_code10.py \
    --model_path=${output_path} \
    --max_new_tokens=400 \
    --batch_size=16 

python evaluate_code.py --predict_file ${output_path}/predictions/humaneval_responses.jsonl

python test_code10.py \
    --model_path=${output_path}/checkpoint-1000 \
    --max_new_tokens=400 \
    --batch_size=16 

python test_code10.py \
    --model_path=${output_path}/checkpoint-400 \
    --max_new_tokens=400 \
    --batch_size=16 

python test_code10.py \
    --model_path=${output_path}/checkpoint-600 \
    --max_new_tokens=400 \
    --batch_size=16 

python test_code10.py \
    --model_path=${output_path}/checkpoint-800 \
    --max_new_tokens=400 \
    --batch_size=16

python evaluate_code.py --predict_file ${output_path}/checkpoint-1000/predictions/humaneval_responses.jsonl
python evaluate_code.py --predict_file ${output_path}/checkpoint-400/predictions/humaneval_responses.jsonl
python evaluate_code.py --predict_file ${output_path}/checkpoint-600/predictions/humaneval_responses.jsonl
python evaluate_code.py --predict_file ${output_path}/checkpoint-800/predictions/humaneval_responses.jsonl
