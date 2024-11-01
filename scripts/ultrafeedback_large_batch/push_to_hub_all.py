import os
import glob
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
    model_dir: str = '/n/holyscratch01/kdbrantley_lab/zhaolin/ultrafeedback_runs/'
    repo_name: str = "GitBag/"
    base_model: str = 'meta-llama/Meta-Llama-3-8B-Instruct'


if __name__ == '__main__':

    args = tyro.cli(Args)

    # init model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
    )
    model = model.cpu()

    for r in glob.glob(args.model_dir + '*_*'):
        repo_name = r.split('/')[-1]
        repo_name = repo_name.split('_')[0] + '_ultrafeedback_' + repo_name.split('_', 1)[1].split('_555134_')[0] + '_' + repo_name.split('_', 1)[1].split('_555134_')[1]
        repo_name = repo_name.replace('=', '_')

        try:
            create_repo(args.repo_name + repo_name)
        except:
            print("Repo created")

        state_dict = get_fp32_state_dict_from_zero_checkpoint(r)
        model.load_state_dict(state_dict)
        model.push_to_hub(args.repo_name + repo_name)
        tokenizer.push_to_hub(args.repo_name + repo_name)

        del state_dict
