#!/bin/bash
#SBATCH --job-name=test_rebel    
#SBATCH -D .
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
# # SBATCH --partition kempner_h100
# # SBATCH --account=kempner_kdbrantley_lab
#SBATCH --partition gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --mem=0
#SBATCH --mail-user=nianli_peng@g.harvard.edu
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

######################
### Set enviroment
######################
#source activateEnvironment.sh
# source ~/.bashrc
# source ~/miniconda3/etc/profile.d/conda.sh
#cd /n/home07/kdbrantley/workspace/rlhf_lite
# conda activate rebel

GPUS_PER_NODE=4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/home03/npeng/anaconda3/lib
container=/n/holyscratch01/kdbrantley_lab/images/pytorch_latest.sif
######################

######################
### Debug Variables
######################
export NCCL_DEBUG=WARN #INFO
export NCCL_SHM_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1 # https://github.com/bigscience-workshop/bigscience/blob/7ccf7e42577fe71e88cf8bed3b9ca965c7afb8f7/train/tr11-176B-ml/tr11-176B-ml.slurm#L207C1-L207C35
export NCCL_P2P_LEVEL=NVL # https://github.com/huggingface/accelerate/issues/314#issuecomment-1565259831
export LOGLEVEL=INFO

######################
### Set Network
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
head_node_port=( $( comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1 ) )

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=$head_node_port
export NUM_PROCESSES=$(expr $SLURM_NNODES \* $GPUS_PER_NODE)

echo "NUM_PROCESSES=$NUM_PROCESSES"

######################
### Set Accelerate
######################
export LAUNCHER="accelerate launch \
    --config_file accelerate_cfgs/deepspeed_config.yaml \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main-process-port $MASTER_PORT \
    --rdzv_backend c10d \
    "
#    --mixed_precision no \
#    --dynamo_backend no \
#    --multi_gpu \

######################
### Set Python
######################
SCRIPT_DIR="/n/home03/npeng/REBEL"

export SCRIPT="${SCRIPT_DIR}/src/tldr/rebel.py"

export SCRIPT_ARGS=" \
    --task.query_dataset GitBag/llama3-ultrafeedback-armo-1024 \
    --task.test_dataset GitBag/llama3-ultrafeedback-armo-1024-test \
    --task.cluster harvard \
    --task.query_length 1024 \
    --task.response_length 1024 \
    --task.temperature 0.8 \
    --lr 3e-7 \
    --warmup_ratio 0.1 \
    --rebel.beta 0.01 \
    --total_episodes 60000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 1 \
    --print_sample_output_freq 200 \
    --exp_name=dpo_armo_OneBatch_newprob_full_lr=3e-7_beta=0.01_bs=128 \
    --output_dir ${SCRIPT_DIR} \
    "
# This step is necessary because accelerate launch does not handle multiline arguments properly
# export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
export CMD="$LAUNCHER $SCRIPT"
srun $CMD

echo "END TIME: $(date)"

# accelerate launch --config_file accelerate_cfgs/ds_config2.yaml --main-process-port 29073 --num_processes 8 src/ultrafeedback_large_batch/simpo.py \
# 	--task.query_dataset GitBag/llama3-ultrafeedback-armo-1024 \
# 	--task.test_dataset GitBag/llama3-ultrafeedback-armo-1024-test \
# 	--task.cluster princeton \
# 	--task.query_length 1024 \
# 	--task.response_length 1024 \
# 	--task.temperature 0.8 \
# 	--lr 1e-6 \
# 	--warmup_ratio 0.1 \
# 	--rebel.beta 10.0 \
# 	--rebel.gamma_beta_ratio 0.3 \
# 	--total_episodes 60000 \
# 	--per_device_train_batch_size 1 \
# 	--gradient_accumulation_steps 16 \
# 	--per_device_eval_batch_size 1 \
# 	--print_sample_output_freq 200 \
# 	--exp_name=simpo_armo_OneBatch_newprob_full_lr=1e-6_beta=10_gb=0.3_bs=128 \
# 	--output_dir /scratch/gpfs/jc9340/ultrafeedback_runs

# accelerate launch --config_file accelerate_cfgs/ds_config2.yaml --main-process-port 29073 --num_processes 8 src/ultrafeedback_large_batch/dpo.py \
# 	--task.query_dataset GitBag/llama3-ultrafeedback-armo-1024 \
# 	--task.test_dataset GitBag/llama3-ultrafeedback-armo-1024-test \
# 	--task.cluster princeton \
# 	--task.query_length 1024 \
# 	--task.response_length 1024 \
# 	--task.temperature 0.8 \
# 	--lr 3e-7 \
# 	--warmup_ratio 0.1 \
# 	--rebel.beta 0.01 \
# 	--total_episodes 60000 \
# 	--per_device_train_batch_size 1 \
# 	--gradient_accumulation_steps 16 \
# 	--per_device_eval_batch_size 1 \
# 	--print_sample_output_freq 200 \
# 	--exp_name=dpo_armo_OneBatch_newprob_full_lr=3e-7_beta=0.01_bs=128 \
# 	--output_dir /scratch/gpfs/jc9340/ultrafeedback_runs