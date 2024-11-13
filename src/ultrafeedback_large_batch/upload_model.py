import tyro
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from dataclasses import dataclass
from huggingface_hub import create_repo


@dataclass
class Args:
    repo_name: str = "Cornell-AGI/REBEL-Llama-3-Armo-iter_3"
    base_model: str = 'GitBag/rebel_ultrafeedback_armo_OneBatch_newprob_full_lr_3e-7_eta_1e2_bs_128_iter_2_1723555887'


if __name__ == '__main__':

    args = tyro.cli(Args)

    try:
        create_repo(args.repo_name)
    except:
        print("Repo created")

    # init model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )

    model = model.cpu()
    tokenizer.push_to_hub(args.repo_name)
    model.push_to_hub(args.repo_name)
