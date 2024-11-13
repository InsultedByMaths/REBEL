import argparse
import torch
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
import pdb

torch.set_printoptions(threshold=10_000)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--reward_model", type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1")
    parser.add_argument("--output_repo", type=str, default="InsultedByMathematics/llama3-ultrafeedback-armo")
    parser.add_argument("--prompts_train", type=str, default="GitBag/llama3-ultrafeedback-armo-1024")
    parser.add_argument("--prompts_test", type=str, default="GitBag/llama3-ultrafeedback-armo-1024-test")
    parser.add_argument("--chosen", type=str, default="bon")
    parser.add_argument("--reject", type=str, default="won")
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument("--maxlen_prompt", type=int, default=1024)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    return parser.parse_args()


def get_message(instruction=None, response=None):

    assert instruction != None or response != None

    if response == None:
        message = [
            {"role": "user", "content": instruction},
        ]
    elif instruction == None:
        message = [
            {"role": "assistant", "content": response}
        ]
    else:
        message = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]

    return message


def filter_same_responses(pairs, row):
    response = row['response_0']
    for i in range(1, pairs):
        if row[f'response_{i}'] != response:
            return True
    return False


def first_true_indices(bools, dtype=torch.long):
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


class ArmoRMPipeline:
    def __init__(self, model_id, device_map="cuda", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return score


class LLMPipeline:
    def __init__(self, args, tokenizer, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=False):
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
            device_map=device_map,
            torch_dtype=torch_dtype,
        ).eval()
        self.tokenizer = tokenizer
        self.device = self.model.device

    def __call__(self, query, response):

        query_response = torch.tensor([query + response]).long().to(self.device)
        response = torch.tensor([response]).long().to(self.device)
        attention_mask = query_response != self.tokenizer.pad_token_id
        input_ids = torch.masked_fill(query_response, ~attention_mask, self.tokenizer.eos_token_id)
        with torch.no_grad():
            output = self.model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        return_dict=True,
                        )
            logits = output.logits[:, self.args.maxlen_prompt - 1 : -1]
            logits /= self.args.temperature + 1e-7
            all_logprob = F.log_softmax(logits, dim=-1)
            logprob = torch.gather(all_logprob, 2, input_ids[:, self.args.maxlen_prompt:].unsqueeze(-1)).squeeze(-1)
            sequence_length = first_true_indices(response == self.tokenizer.pad_token_id) - 1
            seq_mask = torch.arange(self.args.maxlen, device=self.device).unsqueeze(0).expand_as(response) <= sequence_length.unsqueeze(1)
            
            return (logprob * seq_mask).sum(-1).item()


def main(args, dataset_dir, branch):

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer_left = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    tokenizer_left.add_special_tokens({"pad_token": "[PAD]"})

    columns = [f"response_{i}" for i in range(args.pairs)] + ["prompt_id", "prompt", "llama_prompt_tokens"]
    dataset = load_dataset(args.dataset_dir, split=branch)
    dataset = dataset.select_columns(columns)
    # dataset = dataset.select(range(300))

    # ==========================================================================
    # =============================Process Data=================================
    # ==========================================================================
    
    # print('initial length:', len(dataset))

    # # filter dataset with long prompt or response
    # dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(get_message(row['prompt']), tokenize=True, add_generation_prompt=True, return_tensors='pt').shape[-1] <= args.maxlen_prompt)
    # print('filtered long prompts:', len(dataset))
    # for i in range(args.pairs):
    #     dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(get_message(response=row[f'response_{i}']), tokenize=True, add_generation_prompt=False, return_tensors='pt')[:, 5:].shape[-1] <= args.maxlen)
    #     print(f'filtered response_{i}:', len(dataset))

    # # filter prompts with exactly same responses
    # dataset = dataset.filter(lambda row: filter_same_responses(args.pairs, row))
    # print('filtered same responses:', len(dataset))

    # # add prompt tokens
    # llama_prompts = []
    # llama_prompt_tokens = []
    # for row in tqdm(dataset):
    #     llama_prompt_token = tokenizer_left.apply_chat_template(
    #             get_message(row['prompt']), 
    #             add_generation_prompt=True,
    #             tokenize=True,
    #             padding='max_length',
    #             max_length=args.maxlen_prompt,
    #     )
    #     llama_prompt = tokenizer_left.decode(llama_prompt_token, skip_special_tokens=False)
    #     assert len(llama_prompt_token) == args.maxlen_prompt
    #     assert (llama_prompt_token[0] == 128000 or llama_prompt_token[0] == 128256) and llama_prompt_token[-1] == 271
    #     llama_prompts.append(llama_prompt)
    #     llama_prompt_tokens.append(llama_prompt_token)
    # dataset = dataset.add_column("llama_prompt", llama_prompts)
    # dataset = dataset.add_column("llama_prompt_tokens", llama_prompt_tokens)

    # ==========================================================================
    # =============================Gather Reward================================
    # ==========================================================================

    # rewards = {}
    # all_rewards = []
    # rm = ArmoRMPipeline(args.reward_model, trust_remote_code=True)

    # # gather reward
    # for i in range(args.pairs):
    #     print(f'gathering reward for {i+1}th response')
    #     rewards[f"response_{i}_reward"] = []
    #     for row in tqdm(dataset):
    #         reward = rm(get_message(row['prompt'], row[f'response_{i}']))
    #         rewards[f"response_{i}_reward"].append(reward)
    #     all_rewards.append(rewards[f"response_{i}_reward"])
    # for k, v in rewards.items():
    #     dataset = dataset.add_column(k, v)

    # ==========================================================================
    # =============================Gather Logprob===============================
    # ==========================================================================

    llm = LLMPipeline(args, tokenizer, trust_remote_code=True)
    response_tokens, response_logprobs = [], []

    print("\nStarting log probability calculations for each response...")
    # Iterate over each response pair in the dataset
    for i in range(args.pairs):
        print(f"\nCalculating tokens and log probabilities for response_{i}")
        tokens, logprobs = [], []

        # Loop through each dataset entry
        for row in tqdm(dataset, desc=f"Processing logprobs for response_{i}"):
            # Tokenize each response
            response_token = tokenizer.apply_chat_template(
                get_message(response=row[f'response_{i}']),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen + 5,
            )[5:]
            tokens.append(response_token)

            # Ensure the token length matches the maxlen
            assert len(response_token) == args.maxlen, f"Token length mismatch for response_{i}"
            assert response_token[-1] == 128009 or response_token[-1] == 128256, f"Unexpected EOS token for response_{i}"

            # Calculate log probability for each response
            logprob = llm(row["llama_prompt_tokens"], response_token)
            logprobs.append(logprob)

        # Append results for each response to main lists
        response_tokens.append(tokens)
        response_logprobs.append(logprobs)
        print(f"Completed log probability calculation for response_{i}")

    # Add computed columns to the dataset
    print("\nAdding tokens and log probabilities to the dataset...")
    for i in range(args.pairs):
        dataset = dataset.add_column(f"response_{i}_tokens", response_tokens[i])
        dataset = dataset.add_column(f"response_{i}_logprob", response_logprobs[i])
        print(f"Added tokens and log probabilities for response_{i} to the dataset.")

    print("Log probability calculation and dataset update complete.")

    return dataset

if __name__ == "__main__":

    args = parse_arguments()

    train_data = main(args, "prompts_train", "train_prefs")
    print("Training data done.")
    test_data = main(args, "prompts_test", "test_prefs")
    print("Testing data done.")

    data = DatasetDict({
        "train_prefs" : train_data,
        "test_prefs"  : test_data,
    })
    data.push_to_hub(args.output_repo)
    print("Uploading data done.")