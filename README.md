<h1 align="center">FastCLIP: A Suite of Optimization Techniques to <br> Accelerate CLIP Training with Limited Resources</h1>

Existing studies of training state-of-the-art Contrastive Language-Image Pretraining (CLIP) models on large-scale data involve hundreds of or even thousands of GPUs due to the requirement of a large batch size. However, such a large amount of resources is not accessible to most people. While advanced compositional optimization techniques for optimizing global contrastive losses have been demonstrated effective for removing the requirement of large batch size, their performance on large-scale data remains underexplored and not optimized. To bridge the gap, this paper explores several aspects of CLIP training with *limited resources* (e.g., up to tens of GPUs). First, we introduce FastCLIP, a general CLIP training framework built on advanced compositional optimization techniques while designed and optimized for the **distributed setting**. Our framework is equipped with an efficient gradient reduction strategy to reduce communication overhead. Second, to further boost training efficiency, we investigate three components of the framework from an optimization perspective: the schedule of the inner learning rate, the update rules of the temperature parameter and the model parameters, respectively. Experiments on different strategies for each component shed light on how to conduct CLIP training more efficiently. Finally, we benchmark the performance of FastCLIP and the state-of-the-art training baseline (OpenCLIP) on different compute scales up to 32 GPUs on 8 nodes, and three data scales ranging from 2.7 million, 9.1 million to 315 million image-text pairs to demonstrate the significant improvement of FastCLIP in the resource-limited setting.

## Instructions

### Training

For training, please first create an environment using [requirements-training.txt](./requirements-training.txt). The following is a sample slurm script to run FastCLIP-v3 on cc3m using 2 nodes and 4 GPUs per node.
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
**Datacomp**: For evaluation on Datacomp Benchmark, we refer the readers to the `Evaluation` section in the [Datacomp repository](https://github.com/mlfoundations/datacomp?tab=readme-ov-file#evaluation).

**Non-slurm Training**: For non-slurm training, please set `master_addr` manually, change `srun python -u src/training/main.py` to `cd src && torchrun --nproc_per_node=4 --rdzv_endpoint=$master_addr -m training.main`, and run the above script with `/bin/bash`.
