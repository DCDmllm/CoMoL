# CoMoL: Efficient Mixture of LoRA Experts via Dynamic Core Space Merging

## Installation

```bash
# Navigate to the CoMoL directory
cd CoMoL

# Install required dependencies
pip install -r requirements.txt

cd transformers
pip install -e .[torch]
```

## Usage

```bash
# Train the model
python train.py @configs/qwen3-8b_mocorelora_math14k_train_exp8_corerouter.config

# Test the model
python test.py @configs/qwen3-8b_mocorelora_math14k_test_exp8_corerouter.config
```
"mocorelora" denotes CoMoL method in this project.

## Citation
