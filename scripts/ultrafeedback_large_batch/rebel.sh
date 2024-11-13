#!/bin/bash
#SBATCH --job-name=test_rebel    
#SBATCH -D .
#SBATCH -o ./runs/output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./runs/errors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --partition kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
# #SBATCH --partition gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --mem=0
#SBATCH --mail-user=nianli_peng@g.harvard.edu
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --time=0-72:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

######################
### Set enviroment
######################
#source activateEnvironment.sh
# source ~/.bashrc
# source ~/miniconda3/etc/profile.d/conda.sh
#cd /n/home07/kdbrantley/workspace/rlhf_lite

module load cuda

GPUS_PER_NODE=4
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/home03/npeng/anaconda3/lib
container=/n/holyscratch01/kdbrantley_lab/images/pytorch_latest.sif
######################

######################
### Debug Variables
######################
export NCCL_DEBUG=INFO # WARN
export NCCL_SHM_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 # https://github.com/bigscience-workshop/bigscience/blob/7ccf7e42577fe71e88cf8bed3b9ca965c7afb8f7/train/tr11-176B-ml/tr11-176B-ml.slurm#L207C1-L207C35
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
    --dynamo_backend no \
    "
#    --mixed_precision no \
#    --dynamo_backend no \
#    --multi_gpu \

######################
### Set Python
######################
SCRIPT_DIR="/n/home03/npeng/REBEL"

export SCRIPT="${SCRIPT_DIR}/src/ultrafeedback_large_batch/rebel.py"

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