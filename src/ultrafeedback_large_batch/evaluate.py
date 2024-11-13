import os
import tyro
import torch
import argparse
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from dataclasses import dataclass
import deepspeed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import pdb
import math
import time
from typing import Dict, List
import torch.nn.functional as F
import numpy as np
import random

@dataclass
class Args:
    model_dir: str = 'InsultedByMathematics/rebel_ultrafeedback_eta_1e6'
    base_model: str = 'meta-llama/Meta-Llama-3-8B-Instruct'
    reward_model_path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    test_dataset: str = "GitBag/llama3-ultrafeedback-armo-1024-test"
    cluster: str = "harvard"
    maxlen: int = 1024
    maxlen_prompt: int = 1024
    temperature: float = 0.8
    temperature_base: float = 0.8  # Temperature for the base (pretrained) model
    eta: int = 1e6

def set_seed(seed=555134):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    def __init__(self, args, model_dir, tokenizer, temperature, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=False):
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
            device_map=device_map,
            torch_dtype=torch_dtype,
        ).eval()
        self.tokenizer = tokenizer
        self.device = self.model.device
        self.temperature = temperature

    def generate(self, input_ids, attention_mask, generation_config):
        """
        Generates a response based on the provided input IDs and attention mask.
        """
        # Ensure inputs are on the correct device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id  # Ensure pad_token_id is set
        
        # Generate response
        generated_response = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
        return generated_response

    def __call__(self, query, response):
        # Ensure both query and response are on the correct device
        query = query.to(self.device)
        response = response.to(self.device)

        # Concatenate and move to device
        query_response = torch.cat([query[0], response]).unsqueeze(0).long().to(self.device)
        response = response.unsqueeze(0).to(self.device)
        attention_mask = query_response != self.tokenizer.pad_token_id
        input_ids = torch.masked_fill(query_response, ~attention_mask, self.tokenizer.eos_token_id)

        with torch.no_grad():
            # Forward pass to get logits
            output = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask.to(self.device),
                return_dict=True,
            )
            logits = output.logits[:, self.args.maxlen_prompt - 1 : -1]
            logits /= self.temperature + 1e-7

            # Log-softmax for log probabilities
            all_logprob = F.log_softmax(logits, dim=-1)
            logprob = torch.gather(all_logprob, 2, input_ids[:, self.args.maxlen_prompt:].unsqueeze(-1)).squeeze(-1)
            sequence_length = first_true_indices(response == self.tokenizer.pad_token_id) - 1
            try:
                # Calculate sequence length mask
                seq_mask = torch.arange(self.args.maxlen, device=self.device).unsqueeze(0).expand_as(response) <= sequence_length.unsqueeze(1)
            except RuntimeError as e:
                # Print shapes and error details for debugging
                print("Error encountered in calculating `seq_mask`:")
                print(f"Response token: {response}")
                print(f"Response text: {self.tokenizer.decode(response[0], skip_special_tokens=False)}")
                print(f"Response text: {self.tokenizer.decode(response[0], skip_special_tokens=True)}")
                print(f"Response shape: {response.size()}")
                print(f"Sequence length shape: {sequence_length.size()}")
                print(f"Error message: {e}")
            
            return (logprob * seq_mask).sum(-1).item()

if __name__ == '__main__':
    args = tyro.cli(Args)

    print("Arg parsed.")

    # Initialize model and tokenizer
    policy_tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, 
        padding_side='right',
        trust_remote_code=True,
    )
    policy_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Prepare generation configuration
    generation_config = GenerationConfig(
        max_new_tokens=1023, # accounting for eos
        min_new_tokens=1023, # accounting for eos
        temperature=(args.temperature + 1e-7),
        do_sample=True,
    )

    # Load the fine-tuned model with LLMPipeline
    finetuned_model = LLMPipeline(args, args.model_dir, policy_tokenizer, args.temperature, trust_remote_code=True)
    print("Fine-tuned model loaded.")

    # Load the base (pretrained) model with LLMPipeline
    base_model = LLMPipeline(args, args.base_model, policy_tokenizer, args.temperature_base, trust_remote_code=True)
    print("Base model loaded.")

    reward_model = ArmoRMPipeline(args.reward_model_path, trust_remote_code=True)
    print("Reward model loaded.")

    # Load Dataset
    dataset = load_dataset(args.test_dataset + '_' + args.cluster, split='test_prefs')
    dataset = dataset.with_format("torch", columns=["prompt", "llama_prompt_tokens", 
                                                            "llama_chosen_tokens", "chosen_reward", "chosen_logprob",
                                                            "llama_reject_tokens", "reject_reward", "reject_logprob",
                                                            "chosen"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print("Dataset loaded")

    # Initialize lists to store metrics
    queries = []
    responses1 = []
    responses2 = []
    rewards1 = []
    rewards2 = []
    delta_rewards = []
    logprobs1 = []
    logprobs2 = []
    logprobs_base1 = []
    logprobs_base2 = []
    # delta_logprobs = []
    # delta_logprobs_base = []
    delta_ln_pis = []
    chosen_texts = []
    chosen_rewards = []

    for idx, data in enumerate(tqdm(dataloader)):
        query_tokens = data["llama_prompt_tokens"].to(finetuned_model.device)

        # Generate response from policy model using LLMPipeline's generate method
        attention_mask = query_tokens != policy_tokenizer.pad_token_id
        input_ids = torch.masked_fill(query_tokens, ~attention_mask, policy_tokenizer.eos_token_id)

        # Store the query
        queries.append(data["prompt"][0])
        
        # Generate response1
        seed1 = idx * 2  # You can adjust the seed as needed
        set_seed(seed1)
        generated_response1 = finetuned_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
        response_tokens1 = generated_response1[:, query_tokens.shape[1]:]  # trim context
        response_text1 = policy_tokenizer.decode(response_tokens1[0], skip_special_tokens=True)
        responses1.append(response_text1)

        # Generate response2
        seed2 = idx * 2 + 1  # Ensure different seeds
        set_seed(seed2)
        generated_response2 = finetuned_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
        response_tokens2 = generated_response2[:, query_tokens.shape[1]:]  # trim context
        response_text2 = policy_tokenizer.decode(response_tokens2[0], skip_special_tokens=True)
        responses2.append(response_text2)

        # Get reward scores
        reward1 = reward_model(get_message(data['prompt'][0], response_text1))
        reward2 = reward_model(get_message(data['prompt'][0], response_text2))
        torch.cuda.empty_cache()
        rewards1.append(reward1)
        rewards2.append(reward2)
        delta_reward = reward1 - reward2
        delta_rewards.append(delta_reward)

        # Prepare response tokens for log probability calculation
        # response_tokens1_full = policy_tokenizer.apply_chat_template(
        #     get_message(response=response_text1),
        #     add_generation_prompt=False,
        #     tokenize=True,
        #     padding='max_length',
        #     max_length=args.maxlen + 5,
        # )[5:]  # Skip the first 5 tokens (system prompt tokens)

        # response_tokens2_full = policy_tokenizer.apply_chat_template(
        #     get_message(response=response_text2),
        #     add_generation_prompt=False,
        #     tokenize=True,
        #     padding='max_length',
        #     max_length=args.maxlen + 5,
        # )[5:]

        # # policy_tokenizer.apply_chat_template(get_message(response=response_text),add_generation_prompt=False,tokenize=True,padding='max_length',max_length=args.maxlen+5)
        # # df = pd.DataFrame({'response_tokens2_list': response_tokens2_list,'response_tokens2_full': response_tokens2_full[:1023],'difference': difference_values.tolist()})

        # if len(response_tokens1_full) != args.maxlen or len(response_tokens2_full) != args.maxlen:
        #     pdb.set_trace()

        # assert len(response_tokens1_full) == args.maxlen
        # assert len(response_tokens2_full) == args.maxlen

        # response_tokens1_full = torch.tensor(response_tokens1_full)
        # response_tokens2_full = torch.tensor(response_tokens2_full)

        # Prepare response tokens for log probability calculation
        response_tokens1_full = response_tokens1[0]  # Extract tensor from batch
        response_tokens2_full = response_tokens2[0]  # Extract tensor from batch

        # Pad or truncate the response tokens to args.maxlen using tokenizer's pad function
        response_tokens1_padded = policy_tokenizer.pad(
            {"input_ids": response_tokens1_full},
            padding='max_length',
            max_length=args.maxlen,
            return_tensors='pt',
        )["input_ids"].to(finetuned_model.device)

        response_tokens2_padded = policy_tokenizer.pad(
            {"input_ids": response_tokens2_full},
            padding='max_length',
            max_length=args.maxlen,
            return_tensors='pt',
        )["input_ids"].to(finetuned_model.device)

        # pdb.set_trace()

        # Calculate log probabilities under fine-tuned model
        logprob1 = finetuned_model(query_tokens, response_tokens1_padded)
        logprob2 = finetuned_model(query_tokens, response_tokens2_padded)

        # Calculate log probabilities under fine-tuned model
        # logprob1 = finetuned_model(data["llama_prompt_tokens"], response_tokens1_full)
        # logprob2 = finetuned_model(data["llama_prompt_tokens"], response_tokens2_full)
        logprobs1.append(logprob1)
        logprobs2.append(logprob2)
        # delta_logprob = logprob1 - logprob2
        # delta_logprobs.append(delta_logprob)

        # Calculate log probabilities under base (pretrained) model
        # logprob_base1 = base_model(data["llama_prompt_tokens"], response_tokens1_full)
        # logprob_base2 = base_model(data["llama_prompt_tokens"], response_tokens2_full)

        logprob_base1 = base_model(query_tokens, response_tokens1_padded)
        logprob_base2 = base_model(query_tokens, response_tokens2_padded)

        logprobs_base1.append(logprob_base1)
        logprobs_base2.append(logprob_base2)
        # delta_logprob_base = logprob_base1 - logprob_base2
        # delta_logprobs_base.append(delta_logprob_base)

        # Calculate delta ln pi
        delta_ln_pi = ((logprob1 - logprob_base1) - (logprob2 - logprob_base2))
        delta_ln_pis.append(delta_ln_pi)

        # Use the chosen text and reward directly from the dataset
        chosen_texts.append(data["chosen"][0])  
        chosen_rewards.append(data["chosen_reward"].item()) 

        # break

    # Save results to DataFrame
    eval_df = pd.DataFrame({
        "query": queries,
        "response1": responses1,
        "response2": responses2,
        "reward1": rewards1,
        "reward2": rewards2,
        "delta_reward": delta_rewards,
        "logprob1": logprobs1,
        "logprob2": logprobs2,
        # "delta_logprob": delta_logprobs,
        "logprob_base1": logprobs_base1,
        "logprob_base2": logprobs_base2,
        # "delta_logprob_base": delta_logprobs_base,
        "delta_ln_pi": delta_ln_pis,
        "pretrained_response": chosen_texts,
        "pretrained_reward": chosen_rewards
    })
    eval_df.to_csv(f"online_evaluation_results_{int(time.time())}.csv", index=False, escapechar="\\")
    print("Evaluation complete. Results saved to CSV.")