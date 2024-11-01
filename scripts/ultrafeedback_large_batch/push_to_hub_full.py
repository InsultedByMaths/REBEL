import os
import tyro
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from dataclasses import dataclass
from huggingface_hub import HfApi, create_repo, create_branch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


@dataclass
class Args:
    model_dir: str = '/n/holyscratch01/kdbrantley_lab/zhaolin/ultrafeedback_runs/rebel_armo_OneBatch_newprob_full_lr=3e-7_eta=1e3_bs=128_iter_1_555134_1723089445'
    repo_name: str = "GitBag/rebel_ultrafeedback_armo_OneBatch_newprob_full_lr_3e-7_eta_1e6_bs_128_iter_1_1723089445"
    base_model: str = 'meta-llama/Meta-Llama-3-8B-Instruct'


if __name__ == '__main__':

    args = tyro.cli(Args)

    try:
        create_repo(args.repo_name)
    except:
        print("Repo created")

    # init model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
    )

    # model_checkpoints = os.listdir(args.model_dir)

    # for checkpoint in model_checkpoints:

    #     print(checkpoint)

    #     # if checkpoint != '201':
    #     #     continue

    #     try:
    #         create_branch(args.repo_name, branch='step'+checkpoint)
    #     except:
    #         # None
    #         continue

    # load model
    state_dict = get_fp32_state_dict_from_zero_checkpoint(args.model_dir)
    print('loaded')
    model = model.cpu() # move to cpu
    model.load_state_dict(state_dict)
    model.push_to_hub(args.repo_name, revision='main')

    # load tokenizer
    tokenizer.push_to_hub(args.repo_name, revision='main')
