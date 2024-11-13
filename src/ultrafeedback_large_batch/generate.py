from datasets import load_dataset, DatasetDict
from huggingface_hub import create_repo
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import argparse
import torch
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output_repo", type=str, default="GitBag/llama3-ultrafeedback")
    parser.add_argument("--prompts", type=str, default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--world_size", type=int, default=4)
    return parser.parse_args()


def get_message(instruction):
    message = [
        {"role": "user", "content": instruction},
    ]
    return message


def main():

    # init
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.world_size,
    )
    
    # train
    train_data = load_dataset(args.prompts, split="train_prefs")
    if args.end_idx != -1:
        train_data = train_data.select(range(args.start_idx, args.end_idx))

    # construct prompts for the llm and record the prompts & tokens
    prompts = [tokenizer.apply_chat_template(get_message(row['prompt']), tokenize=False, add_generation_prompt=True) for row in tqdm(train_data)]
    train_data = train_data.add_column("processed_prompt", prompts)

    # start generate
    for p in range(args.pairs):
        set_seed(p * 50)
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=args.maxlen,
            seed=p * 50,
        )
        response = llm.generate(prompts, sampling_params)
        output = list(map(lambda x: x.outputs[0].text, response))
        train_data = train_data.add_column(f"response_{p}", output)

    # test
    test_data = load_dataset(args.prompts, split="test_prefs")

    # construct prompts for the llm and record the prompts & tokens
    prompts = [tokenizer.apply_chat_template(get_message(row['prompt']), tokenize=False, add_generation_prompt=True) for row in tqdm(test_data)]
    test_data = test_data.add_column("processed_prompt", prompts)

    # start generate
    for p in range(args.pairs):
        set_seed(p * 50)
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=args.maxlen,
            seed=p * 50,
        )
        response = llm.generate(prompts, sampling_params)
        output = list(map(lambda x: x.outputs[0].text, response))
        test_data = test_data.add_column(f"response_{p}", output)

    data = DatasetDict({
        "train_prefs" : train_data,
        "test_prefs"  : test_data,
    })
    data.push_to_hub(args.output_repo)


if __name__ == "__main__":
    main()