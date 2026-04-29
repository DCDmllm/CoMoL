export CUDA_VISIBLE_DEVICES="0"

# Count the number of devices
num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Number of devices: $num_devices"

base_model_path=Qwen/Qwen3-8B
base_model=qwen3-8b
dataset_path=./datasets/math_14k
dataset=math-14k
model=mole
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
    --top_k=2 \
    --max_length=300 \
    --batch_size=4 \
    --gradient_accumulation_steps=4 \
    --num_train_epochs=1 \
    --learning_rate=1e-4 \
    --lr_scheduler_type=constant_with_warmup \
    --warmup_steps=200 \
    --weight_decay=0.0 \
    --aux_loss_coeff=1e-3

output_path=outputs/${base_model}-${model}-top2-rank${rank}-exp${experts}-${dataset}

python test_math.py \
    --model_path=${output_path} \
    --max_new_tokens=300 \
    --batch_size=64

python evaluate_math.py --predict_file ${output_path}/predictions/addsub_responses.jsonl