# CoMoL: Efficient Mixture of LoRA Experts via Dynamic Core Space Merging

This repository contains the official implementation of **CoMoL**, accepted to **ACL 2026 Findings**.

CoMoL is a parameter-efficient fine-tuning method for large language models. Instead of assigning each expert a full LoRA branch, CoMoL places the mixture in a compact **core space** between the LoRA down-projection and up-projection. A lightweight router dynamically merges multiple learnable core matrices for each token, enabling a mixture-of-experts style adapter while keeping the trainable parameter cost close to LoRA.

In this codebase, the paper method is implemented under the name `mocorelora`.

## Highlights

- Implements CoMoL / `mocorelora` for causal language model fine-tuning.
- Includes comparison PEFT methods: `lora`, `mole`, `adamole`, `denselora`, `molora`, `hydralora`, and `flylora`.
- Provides training and evaluation scripts for math reasoning, commonsense reasoning, code generation.
- Builds on the Hugging Face `transformers`, `peft` library.

## Repository Layout

```text
CoMoL/
├── train.py                     # Main fine-tuning entry point
├── data.py                      # Dataset loading and instruction formatting
├── test_math.py                 # Generation for math / commonsense benchmarks
├── evaluate_math.py             # Accuracy evaluation for math benchmarks
├── test_code.py                 # Generation for HumanEval code tasks
├── evaluate_code.py             # HumanEval functional correctness evaluation
├── src/
│   ├── mocorelora/              # CoMoL implementation
│   ├── lora/                    # LoRA baseline
│   ├── mole/                    # MoLE baseline
│   ├── adamole/                 # AdaMoLE baseline
│   ├── denselora/               # DenseLoRA baseline
│   ├── molora/                  # MoLoRA / HydraLoRA baseline
│   ├── flylora/                 # FlyLoRA baseline
│   ├── peft_model.py            # PEFT model wrapper
│   └── trainer.py               # Trainer with auxiliary/core losses
├── datasets/                    # Training and evaluation datasets
├── exps/math14k/                # Public example scripts
└── requirements.txt             # Python dependencies
```

## Installation

```bash
# create comol enviroment
conda create -n comol python==3.10

# Navigate to the CoMoL directory
cd CoMoL

# Install required dependencies
pip install -r requirements.txt

```
## Quick Start

The easiest way to reproduce the public math example is:

```bash
bash ./exps/math14k/finetune_qwen_mocorelora_corerouter_exp8.sh
```

This script fine-tunes `Qwen/Qwen3-8B` on `datasets/math_14k` with CoMoL, 8 experts, rank 16, and core-space routing. It then runs math benchmark generation and evaluates the produced predictions.

## Training

You can also call `train.py` directly:

```bash
python train.py \
  --model_path Qwen/Qwen3-8B \
  --data_path ./datasets/math_14k \
  --output_dir outputs \
  --peft_type mocorelora \
  --lora_rank 16 \
  --target_modules q_proj k_proj v_proj o_proj down_proj \
  --num_experts 8 \
  --core_router True \
  --max_length 300 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --warmup_steps 200
```

Important arguments:

| Argument | Description |
| --- | --- |
| `--peft_type` | Adapter type. Use `mocorelora` for CoMoL. |
| `--lora_rank` | LoRA rank used by most methods. |
| `--target_modules` | Transformer modules to adapt, e.g. `q_proj k_proj v_proj o_proj down_proj`. |
| `--num_experts` | Number of experts for mixture-based methods. |
| `--core_router` | Route using the LoRA core representation for CoMoL variants. |
| `--aux_loss_coeff` | Weight for auxiliary losses used by the custom trainer. |
| `--top_k`, `--threshold` | Gating controls used by selected baseline methods. |

Checkpoints are saved under `outputs/` with an automatically generated name that includes the base model, PEFT type, target modules, rank, expert count, seed, and dataset name.

## Evaluation

### Math Reasoning

```bash
python test_math.py \
  --model_path outputs/qwen3-8b-mocorelora-corerouter-qkvodown-rank16-exp8-math-14k \
  --data_path ./datasets/math_commonsense \
  --max_new_tokens 300 \
  --batch_size 64

python evaluate_math.py \
  --predict_file outputs/qwen3-8b-mocorelora-corerouter-qkvodown-rank16-exp8-math-14k/predictions/addsub_responses.jsonl
```

`test_math.py` evaluates all math subsets when the model path contains `math`: `AddSub`, `AQuA`, `gsm8k`, `MultiArith`, `SingleEq`, and `SVAMP`.


### Code Generation

```bash
python test_code.py \
  --model_path path/to/checkpoint \
  --data_path ./datasets/eval_code \
  --max_new_tokens 400 \
  --batch_size 64

python evaluate_code.py \
  --predict_file path/to/checkpoint/predictions/humaneval_responses.jsonl
```


## Method Names

| `--peft_type` | Method |
| --- | --- |
| `mocorelora` | CoMoL, the main method in the paper |
| `lora` | Standard LoRA baseline |
| `mole` | Mixture of LoRA Experts baseline |
| `adamole` | Adaptive MoLE baseline |
| `molora` | MoLoRA baseline; `--hydra=True` enables the HydraLoRA setting |
| `denselora` | DenseLoRA baseline |
| `flylora` | FlyLoRA baseline |

## Public Example Scripts

The public `exps/math14k/` directory includes runnable examples for:

- CoMoL with Qwen3-8B and Qwen3-14B.
- CoMoL with 8 or 64 experts.
- LoRA, AdaMoLE, MoLE, MoLoRA, HydraLoRA, and FlyLoRA baselines.

For new experiments, start from one of these scripts and adjust the base model, target modules, rank, number of experts, batch size, and output path.

## Citation

```
@article{cao2026comol,
  title={CoMoL: Efficient Mixture of LoRA Experts via Dynamic Core Space Merging},
  author={Cao, Jie and Fan, Zhenxuan and Wang, Zhuonan and Lin, Tianwei and Zhao, Ziyuan and Yan, Rolan and Zhang, Wenqiao and Shao, Feifei and Wang, Hongwei and Xiao, Jun and others},
  journal={arXiv preprint arXiv:2603.00573},
  year={2026}
}
```
