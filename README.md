# CoMoL: Efficient Mixture of LoRA Experts via Dynamic Core Space Merging

## Installation

```bash
# create comol enviroment
conda create -n comol python==3.10

# Navigate to the CoMoL directory
cd CoMoL

# Install required dependencies
pip install -r requirements.txt

```

## Usage

```bash
# Train & Evaluate the model
bash ./exps/math14k/finetune_qwen_mocorelora_corerouter_exp8.sh
```
"mocorelora" denotes CoMoL method in this project.

## Citation
```
@article{cao2026comol,
  title={CoMoL: Efficient Mixture of LoRA Experts via Dynamic Core Space Merging},
  author={Cao, Jie and Fan, Zhenxuan and Wang, Zhuonan and Lin, Tianwei and Zhao, Ziyuan and Yan, Rolan and Zhang, Wenqiao and Shao, Feifei and Wang, Hongwei and Xiao, Jun and others},
  journal={arXiv preprint arXiv:2603.00573},
  year={2026}
}
```
