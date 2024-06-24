<h1 align="center">FastCLIP: A Suite of Optimization Techniques to <br> Accelerate CLIP Training with Limited Resources</h1>

**TL;DR**: We propose FastCLIP, a CLIP training framework that i) does not require a large batch size to achieve good performance (limited-resource setting), and ii) is more communication-efficient than OpenCLIP. We investigate three optimization componentes of FastCLIP and compare different strategies for each component. Finally, we conduct experiments on CC3M, CC12M and LAION400M to demonstrate the effectiveness of FastCLIP.

## Introduction

### The Proposed FastCLIP Framework

Vanilla mini-batch based methods for self-supervised contrastive learning (e.g., CLIP) are known to require a large batch size to obtain satisfactory performance. Recently, the SogCLR algorithm is proposed to address the large batch size issue, which leverages **finite-sum coupled compositional optimization (FCCO)** techniques. A key feature of compositional optimization is the **inner and outer steps** where the inner steps maintain and update a sequence of estimators to track the inner functions on the solution path, which can be interpreted as an SGD update with a learning rate called the inner learning rate.

In order to scale up the advanced optimization algorithms for optimizing global contrastive losses of CLIP training on large-scale data with limited compute resources, we introduce FastCLIP, a distributed training framework in the data-parallel setting. The algorithmic design is based on SogCLR, and the implementation is based on OpenCLIP. A novel gradient reduction strategy is designed so that it requires less communication than OpenCLIP. This distributed training framework lays the foundation for scaling up CLIP training with limited resources.

To further boost the efficiency of our framework, we investigate its three aspects from an optimization perspective: the schedule of the inner learning rate (LR) of compositional optimization, the update rule of the temperature parameter, and the update rule of the model parameters, respectively.

Moreover, we compare the performance of FastCLIP and OpenCLIP on three data scales and four compute scales. The data scales include 2.7 million (CC3M), 9.1 million (CC12M), and 315 million (LAION400M) image-text pairs (our downloaded versions of these datasets are smaller than their original versions because some web links are not valid anymore). The compute scales include 1, 2, 4, and 8 nodes, with 4 GPUs on each node.

### Experiment Results

Here we only present part of the results of OpenCLIP vs. FastCLIP-v3, which is one of several algorithms in the FastCLIP framework. For more results OpenCLIP vs. FastCLIP-v3, and results of different optimization components of FastCLIP, please refer to our paper. The following figure is the average of ImageNet and its variants (ImageNet-Sketch, ImageNet-v2, ImageNet-A, ImageNet-O, ImageNet-R and ObjectNet) curves of OpenCLIP and FastCLIP-v3 in the medium-scale (CC3M, batch size 1024) and large-scale (CC12M, batch size 2048) settings. From the results we can see that FastCLIP-v3 has a significant improvement and speedup over OpenCLIP.

<p align="center"><img alt="OpenCLIP vs. FastCLIP-v3" src="./assets/openclip_fastclipv3_in_variants_curve.png" width="600"/></p>

In the following figure, (a) and (b) are the average of ImageNet and its variants of OpenCLIP and FastCLIP-v3 across different number of nodes in the medium-scale and large-scale settings, respectively. While (c) is the ImageNet Top1 accuracy of OpenCLIP and FastCLIP-v3 in the xlarge-scale setting (LAION400M, batch size 5120). From the results we can see that FastCLIP-v3 outperforms OpenCLIP by a large margin. Moreover, from (a) and (b) we observe that the performance of FastCLIP-v3 plateaus at 2 nodes, which verifies that FastCLIP does not require a large amount of computing resources.

<p align="center"><img alt="OpenCLIP vs. FastCLIP-v3, Scaling performance" src="./assets/openclip_fastclipv3_in_variants_nodes.png" width="600"/></p>

Besides performance on downstream tasks, we also compare training time of OpenCLIP and FastCLIP. The following figure shows the training time of OpenCLIP and three algorithms in the FastCLIP framework in the medium-scale and large-scale settings. Subfigures (a) and (b) plot the per-iteration training time. Each bar is divided into three parts (from top to bottom): computation, pure communication (not overlapped with computation), and others. Subfigures (c) and (d) plot the communication time per iteration. Each bar is divided into two parts (from top to bottom): communication overlapped with computation and pure communication. We can see that FastCLIP has a shorter communication time, which demonstrates the effectiveness of our efficient gradient computation/communication strategy.

<p align="center"><img alt="OpenCLIP vs. FastCLIP, Training time" src="./assets/openclip_fastclip_time_nodes.png" width="600"/></p>

## Getting Started

### Environment Setup

To set up the environment for training, please
1. Download this repository:
    ```bash
    git clone https://github.com/Optimization-AI/fast_clip.git
    cd fast_clip
    ```
2. Create a new environment:
    ```bash
    conda create -n fastclip python=3.11
    conda activate fastclip
    pip install -r requirements-training.txt
    ```

### Training

We present sample scripts to run OpenCLIP and FastCLIP-v0 to v3 using slurm. For non-slurm instructions, please refer to the end of this subsection. The following is a sample slurm script to run FastCLIP-v3 on cc3m using 2 nodes and 4 GPUs per node.
```bash
#!/bin/bash -x
#SBATCH --time=2-00:00:00
#SBATCH --mem=120G
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=fastclipv3
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.log

source ~/.bashrc
conda activate fastclip

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12805

export CUDA_VISIBLE_DEVICES='0,1,2,3'
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

srun python -u src/training/main.py \
    --save-frequency 1 \
    --train-data './datasets/cc3m_webdataset/cc3m_train/{00000..00331}.tar' \
    --train-num-samples 2723840 --data_size 3318333 \
    --warmup 10000 \
    --batch-size 256 \
    --epochs 37 \
    --workers 6 \
    --model RN50 \
    --name medium_fastclipv3 \
    --seed 2024 \
    --profile \
    --wd 0.1 \
    --local-loss \
    --fastclip --multiply_tau --temperature_scheme global_learnable \
    --lr 1e-3 --lr_tau 2e-4 --lr_tau_scheduler step_thresh --rho 6.5 \
    --gamma 0.2 --gamma_schedule cosine --gamma_decay_epochs 18
```

To run OpenCLIP, replace the `srun python -u src/training/main.py` command with
```bash
srun python -u src/training/main.py \
    --save-frequency 1 \
    --train-data './datasets/cc3m_webdataset/cc3m_train/{00000..00331}.tar' \
    --train-num-samples 2723840 --data_size 3318333 \
    --warmup 10000 \
    --batch-size 256 \
    --epochs 37 \
    --workers 6 \
    --model RN50 \
    --name medium_openclip \
    --seed 2024 \
    --profile \
    --wd 0.1 \
    --local-loss \
    --gather-with-grad \
    --lr 1e-3
```

To run FastCLIP-v0, replace the `srun python -u src/training/main.py` command with
```bash
srun python -u src/training/main.py \
    --save-frequency 1 \
    --train-data './datasets/cc3m_webdataset/cc3m_train/{00000..00331}.tar' \
    --train-num-samples 2723840 --data_size 3318333 \
    --warmup 10000 \
    --batch-size 256 \
    --epochs 37 \
    --workers 6 \
    --model RN50 \
    --name medium_openclip \
    --seed 2024 \
    --profile \
    --wd 0.1 \
    --local-loss \
    --fastclip --temperature_scheme global_learnable \
    --lr 1e-3 \
    --gamma 0.2 --gamma_schedule cosine --gamma_decay_epochs 18
```

To run FastCLIP-v1, replace the `srun python -u src/training/main.py` command with
```bash
srun python -u src/training/main.py \
    --save-frequency 1 \
    --train-data './datasets/cc3m_webdataset/cc3m_train/{00000..00331}.tar' \
    --train-num-samples 2723840 --data_size 3318333 \
    --warmup 10000 \
    --batch-size 256 \
    --epochs 37 \
    --workers 6 \
    --model RN50 \
    --name medium_openclip \
    --seed 2024 \
    --profile \
    --wd 0.1 \
    --local-loss \
    --fastclip --temperature_scheme global_constant \
    --lr 1e-3 \
    --gamma 0.2 --gamma_schedule cosine --gamma_decay_epochs 18
```

To run FastCLIP-v2, replace the `srun python -u src/training/main.py` command with
```bash
srun python -u src/training/main.py \
    --save-frequency 1 \
    --train-data './datasets/cc3m_webdataset/cc3m_train/{00000..00331}.tar' \
    --train-num-samples 2723840 --data_size 3318333 \
    --warmup 10000 \
    --batch-size 256 \
    --epochs 37 \
    --workers 6 \
    --model RN50 \
    --name medium_openclip \
    --seed 2024 \
    --profile \
    --wd 0.1 \
    --local-loss \
    --fastclip --temperature_scheme individual_learnable \
    --lr 1e-3 --lr_tau 0.0133 --lr_tau_scheduler const --temperature 0.03 --rho 7.0 \
    --gamma 0.2 --gamma_schedule cosine --gamma_decay_epochs 18
```
**Non-slurm Training**: For non-slurm training, please set `master_addr` manually, change `srun python -u src/training/main.py` to `cd src && torchrun --nproc_per_node=4 --rdzv_endpoint=$master_addr -m training.main`, and run the above script with `/bin/bash`.

### Evaluation

**ImageNet-1k**: The following is a sample slurm script to evaluate a trained CLIP model (specified by `--resume`) on ImageNet-1k.
```bash
#!/bin/bash -x
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=eval_fastclip
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.log

source ~/.bashrc
conda activate fastclip

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12802

export PYTHONPATH="$PYTHONPATH:$PWD/src"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

srun python -u src/training/main.py \
    --resume ./logs/medium_fastclipv3/checkpoints/epoch_37.pt \
    --zeroshot-frequency 1 \
    --imagenet-val ./datasets/imagenet/val \
    --batch-size 512 \
    --epochs 0 \
    --workers 6 \
    --model RN50 \
    --name eval_medium_fastclipv3_epoch_37 \
    --seed 2024
```
**Datacomp**: For evaluation on Datacomp Benchmark, please refer to the `Evaluation` section in the [Datacomp repository](https://github.com/mlfoundations/datacomp?tab=readme-ov-file#evaluation).

**Non-slurm Training**: For non-slurm training, please set `master_addr` manually, change `srun python -u src/training/main.py` to `python src/training/main.py`, and run the above script with `/bin/bash`.
