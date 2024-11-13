import os
import random
import time
import math
from dataclasses import asdict, dataclass, field
from functools import partial
from types import SimpleNamespace
from typing import List, Literal, Optional, Tuple, Union
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb
import deepspeed
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from torch.optim.lr_scheduler import LambdaLR
from huggingface_hub import snapshot_download
from datetime import timedelta

# torch.set_printoptions(threshold=10_000)
# os.environ['NCCL_BLOCKING_WAIT'] = '0'


@dataclass
class REBELHParams:
    num_updates: tyro.conf.Suppress[int] = 468
    whiten_rewards: bool = False
    shift_mean: bool = False
    eta: float = 1e6


@dataclass
class TaskHParams:
    query_length: int = 1024
    query_dataset: str = "GitBag/llama3-ultrafeedback-armo-1024"
    test_dataset: str = "GitBag/llama3-ultrafeedback-armo-1024-test"
    cluster: str = "harvard"
    response_length: int = 1024
    temperature: float = 0.8


@dataclass
class Args:
    # common args
    exp_name: str = "ultrafeedback_infonca"
    """the name of this experiment"""
    seed: int = 555134
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ultrafeedback"
    """the wandb's project name"""
    run_name: Optional[str] = None
    """a unique name of this run"""
    print_sample_output_freq: int = 200
    """How often to print sample output"""

    # optimizer args
    eps: float = 1e-8
    """the epsilon value for the optimizer"""
    lr: float = 3e-7
    """learning rate"""
    weight_decay: float = 1e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    warmup_ratio: float = 0.1
    """warmup ratio"""

    gradient_accumulation_steps: int = 16
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: int = 1
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    per_device_eval_batch_size: int = 1
    """per rank eval batch size"""
    total_episodes: int = 60000
    """The total number of episodes in the dataset"""

    # optional args filled while running
    world_size: Optional[int] = 8
    """The number of processes (GPUs) to use"""
    batch_size: Optional[int] = 128
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_batch_size: Optional[int] = 16
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""

    # other args 
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    """the name of the pretrained model to use"""
    output_dir: str = "/n/holyscratch01/kdbrantley_lab/npeng/ultrafeedback_runs"
    """Where to save the model"""
    # Number of responses per prompt
    num_responses: int = 5
    task: TaskHParams = field(default_factory=TaskHParams)
    rebel: REBELHParams = field(default_factory=REBELHParams)


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def gather_logprob(args, model, tokenizer, query, response, device):

    query_response = torch.cat((query, response), dim=-1).long().to(device).unsqueeze(0)
    response = response.long().to(device).unsqueeze(0)
    attention_mask = query_response != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_response, ~attention_mask, tokenizer.eos_token_id)
    with torch.no_grad():
        output = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    return_dict=True,
                 )
        logits = output.logits[:, args.task.query_length - 1 : -1]
        logits /= args.task.temperature + 1e-7
        all_logprob = F.log_softmax(logits, dim=-1)
        logprob = torch.gather(all_logprob, 2, input_ids[:, args.task.query_length:].unsqueeze(-1)).squeeze(-1)
        sequence_length = first_true_indices(response == tokenizer.pad_token_id) - 1
        seq_mask = torch.arange(args.task.response_length, device=device).unsqueeze(0).expand_as(response) <= sequence_length.unsqueeze(1)
        
        return (logprob * seq_mask).sum(-1).item()


def evaluate(args, policy, tokenizer, dataloader):

    device = policy.device
    loss, loss_eta_1, sign_align = [], [], []
    with torch.no_grad():
        for data in tqdm(dataloader):
            
            responses = torch.cat((data["llama_chosen_tokens"], data["llama_reject_tokens"]), dim=0)
            logprobs = torch.cat((data["chosen_logprob"], data["reject_logprob"]), dim=0)
            query_responses = torch.cat((torch.cat((data["llama_prompt_tokens"], data["llama_prompt_tokens"]), dim=0), responses), dim=1)
            sequence_length = first_true_indices(responses == tokenizer.pad_token_id) - 1
            seq_mask = torch.arange(args.task.response_length, device=device).unsqueeze(0).expand_as(responses) <= sequence_length.unsqueeze(1)

            attention_mask = query_responses != tokenizer.pad_token_id
            input_ids = torch.masked_fill(query_responses, ~attention_mask, tokenizer.eos_token_id)

            output = policy(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            logits = output.logits[:, args.task.query_length - 1 : -1]
            logits /= args.task.temperature + 1e-7
            new_all_logprobs = F.log_softmax(logits, dim=-1)
            new_logprobs = torch.gather(new_all_logprobs, 2, input_ids[:, args.task.query_length:].unsqueeze(-1)).squeeze(-1)
            new_logprobs = (new_logprobs * seq_mask).sum(-1)
            ratio_logprob = new_logprobs - logprobs
            ratio_logprob = ratio_logprob[:args.per_device_eval_batch_size] - ratio_logprob[args.per_device_eval_batch_size:]

            reg_diff = ratio_logprob - args.rebel.eta * (data["chosen_reward"] - data["reject_reward"])
            loss.append((reg_diff ** 2).mean().reshape(1))

            reg_diff = ratio_logprob - (data["chosen_reward"] - data["reject_reward"])
            loss_eta_1.append((reg_diff ** 2).mean().reshape(1))

            sign_align.append((ratio_logprob > 0).float().mean().reshape(1))

    loss = torch.cat(loss)
    loss_eta_1 = torch.cat(loss_eta_1)
    sign_align = torch.cat(sign_align)
    return {"val_loss" : loss, "val_loss_eta=1" : loss_eta_1, "sign_align" : sign_align}

# Modify data loading to handle multiple responses
def load_multiple_response_dataset(args, dataset_name):
    dataset = load_dataset(dataset_name, split='train_prefs')
    # Dynamically load response tokens, rewards, and logprobs
    response_columns = [f"response_{i}_tokens" for i in range(args.num_responses)]
    reward_columns = [f"response_{i}_reward" for i in range(args.num_responses)]
    logprob_columns = [f"response_{i}_logprob" for i in range(args.num_responses)]

    columns = ["llama_prompt_tokens"] + response_columns + reward_columns + logprob_columns
    dataset = dataset.with_format("torch", columns=columns)
    return dataset

# Updated loss calculation function for InfoNCA
def infoNCA_loss(args, policy, tokenizer, mb_query, mb_responses, mb_rewards, device):
    """
    Calculates the InfoNCA loss over multiple responses per query in a mini-batch.
    """
    num_responses = len(mb_responses)
    mb_query_responses = torch.cat([torch.cat((mb_query, response), dim=1).unsqueeze(1) for response in mb_responses], dim=1)  # (batch_size, num_responses, total_seq_len)
    
    attention_mask = mb_query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(mb_query_responses, ~attention_mask, tokenizer.eos_token_id)

    with torch.no_grad():
        output = policy(
            input_ids=input_ids.view(-1, input_ids.size(-1)),  # Flatten batch and num_responses dimensions
            attention_mask=attention_mask.view(-1, attention_mask.size(-1)),
            return_dict=True
        )
        logits = output.logits.view(input_ids.size(0), num_responses, -1, logits.size(-1))  # (batch_size, num_responses, seq_len, vocab_size)
        logits /= args.task.temperature + 1e-7
        all_logprobs = F.log_softmax(logits, dim=-1)

    # Calculate log-probabilities for each response by gathering the relevant logits
    model_logprobs = torch.stack(
        [torch.gather(all_logprobs[:, i], 2, input_ids[:, i].unsqueeze(-1)).squeeze(-1).sum(dim=-1) for i in range(num_responses)],
        dim=1
    )

    # Softmax over the true rewards to get the target distribution
    soft_labels = F.softmax(torch.stack(mb_rewards, dim=1) / args.rebel.eta, dim=-1)

    # InfoNCA loss: cross-entropy between model log-probabilities and true reward distribution
    loss = -torch.sum(soft_labels * model_logprobs, dim=-1).mean()

    return loss

if __name__ == '__main__':

    args = tyro.cli(Args)
    # process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours
    # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[process_group_kwargs])
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

    args.world_size = accelerator.num_processes
    args.batch_size = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if args.rebel.whiten_rewards:
        assert (args.local_batch_size >= 8), f"Per-rank minibatch size {args.local_batch_size} is insufficient for whitening"
    args.rebel.num_updates = args.total_episodes // args.batch_size

    # logging
    console = Console(force_terminal=True)
    accelerator.wait_for_everyone()
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
    accelerator.print("Wandb run name: ", run_name)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z, max_bins: None
    if accelerator.is_main_process:
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                sync_tensorboard=True,
                config=asdict(args),
                name=run_name,
                save_code=True,
            )
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    torch.backends.cudnn.deterministic = True

    # policy
    tokenizer = AutoTokenizer.from_pretrained(
                    args.base_model, 
                    padding_side='right',
                    trust_remote_code=True,
                )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    policy = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
    disable_dropout_in_model(policy)

    # Prompt Collection Dataset
    recompute_log = False
    try:
        # dataset = load_dataset(args.task.query_dataset + '_' + args.task.cluster, split='train_prefs')
        # dataset = dataset.with_format("torch", columns=["llama_prompt_tokens", 
        #                                                 "llama_chosen_tokens", "chosen_reward", "chosen_logprob",
        #                                                 "llama_reject_tokens", "reject_reward", "reject_logprob"])
        # Load the dataset with multiple responses
        dataset = load_multiple_response_dataset(args, args.task.query_dataset)
        temp_dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
        validation_dataset = load_dataset(args.task.test_dataset + '_' + args.task.cluster, split='test_prefs')
        validation_dataset = validation_dataset.with_format("torch", columns=["llama_prompt_tokens", 
                                                            "llama_chosen_tokens", "chosen_reward", "chosen_logprob",
                                                            "llama_reject_tokens", "reject_reward", "reject_logprob"])
    except:
        dataset = load_dataset(args.task.query_dataset, split='train_prefs')
        dataset = dataset.with_format("torch", columns=["llama_prompt_tokens", 
                                                        "llama_chosen_tokens", "chosen_reward", "chosen_logprob",
                                                        "llama_reject_tokens", "reject_reward", "reject_logprob"])
        temp_dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
        validation_dataset = load_dataset(args.task.test_dataset, split='test_prefs')
        validation_dataset = validation_dataset.with_format("torch", columns=["llama_prompt_tokens", 
                                                            "llama_chosen_tokens", "chosen_reward", "chosen_logprob",
                                                            "llama_reject_tokens", "reject_reward", "reject_logprob"])
        recompute_log = True

    if accelerator.is_main_process:
        pprint(policy.config)

    if args.optimizer == "adam":
        optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            policy.parameters(), 
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=args.eps,
            weight_decay=args.weight_decay
        )
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.rebel.num_updates * args.warmup_ratio * args.world_size), args.rebel.num_updates * args.world_size)

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    policy, optimizer, _, scheduler = accelerator.prepare(policy, optimizer, temp_dataloader, scheduler)

    if recompute_log:
        accelerator.print('gathering validation logprob')
        chosen_logprob, reject_logprob = [], []
        for i in tqdm(range(len(validation_dataset))):
            chosen_logprob.append(gather_logprob(args, accelerator.unwrap_model(policy), tokenizer, validation_dataset[i]["llama_prompt_tokens"], validation_dataset[i]["llama_chosen_tokens"], device))
            reject_logprob.append(gather_logprob(args, accelerator.unwrap_model(policy), tokenizer, validation_dataset[i]["llama_prompt_tokens"], validation_dataset[i]["llama_reject_tokens"], device))
        validation_dataset = validation_dataset.remove_columns(["chosen_logprob", "reject_logprob"])
        validation_dataset = validation_dataset.add_column("chosen_logprob", chosen_logprob)
        validation_dataset = validation_dataset.add_column("reject_logprob", reject_logprob)
        validation_dataset = validation_dataset.with_format("torch", columns=["llama_prompt_tokens", 
                                                                              "llama_chosen_tokens", "chosen_reward", "chosen_logprob",
                                                                              "llama_reject_tokens", "reject_reward", "reject_logprob"])
        if accelerator.is_main_process:
            validation_dataset.push_to_hub(args.task.test_dataset + '_' + args.task.cluster)

        accelerator.print('gathering logprob')
        chosen_logprob, reject_logprob = [], []
        for i in tqdm(range(len(dataset))):
            chosen_logprob.append(gather_logprob(args, accelerator.unwrap_model(policy), tokenizer, dataset[i]["llama_prompt_tokens"], dataset[i]["llama_chosen_tokens"], device))
            reject_logprob.append(gather_logprob(args, accelerator.unwrap_model(policy), tokenizer, dataset[i]["llama_prompt_tokens"], dataset[i]["llama_reject_tokens"], device))
        dataset = dataset.remove_columns(["chosen_logprob", "reject_logprob"])
        dataset = dataset.add_column("chosen_logprob", chosen_logprob)
        dataset = dataset.add_column("reject_logprob", reject_logprob)
        dataset = dataset.with_format("torch", columns=["llama_prompt_tokens", 
                                                        "llama_chosen_tokens", "chosen_reward", "chosen_logprob",
                                                        "llama_reject_tokens", "reject_reward", "reject_logprob"])
        if accelerator.is_main_process:
            dataset.push_to_hub(args.task.query_dataset + '_' + args.task.cluster)

    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)
    dataloader = accelerator.prepare(dataloader)
    validation_dataloader = accelerator.prepare(validation_dataloader)
    def repeat_generator():
        while True:
            yield from dataloader
    iter_dataloader = iter(repeat_generator())

    accelerator.print("===training policy===")
    torch.manual_seed(local_seed)  # reset the local seed again
    global_step = 0
    start_time = time.time()

    kl_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    chosen_kl_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    reject_kl_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    loss_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    ratio_stats = torch.zeros(args.gradient_accumulation_steps, device=device)

    policy.train()
    # Modified training loop for InfoNCA with mini-batch gradient accumulation
    for update in range(1, args.rebel.num_updates + 1):

        # update parameters
        global_step += 1 * args.batch_size
        lrnow = optimizer.param_groups[0]["lr"]

        # save model
        if (update - 1) % args.print_sample_output_freq == 0: # !!!!!!! and update > 1
            eval_dict = evaluate(args, accelerator.unwrap_model(policy), tokenizer, validation_dataloader)
            writer.add_scalar("objective/validation_loss", accelerator.gather(eval_dict["val_loss"]).mean().item(), update)
            writer.add_scalar("objective/validation_loss_eta=1", accelerator.gather(eval_dict["val_loss_eta=1"]).mean().item(), update)
            writer.add_scalar("objective/sign_align", accelerator.gather(eval_dict["sign_align"]).mean().item(), update)
            if args.output_dir:
                accelerator.wait_for_everyone()
                output_dir = os.path.join(args.output_dir, run_name) # !!!!!!!! str(update)
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                accelerator.save_state(output_dir=output_dir)
                accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

        # training
        data = next(iter_dataloader) # Get next batch from iterator

        # Initialize gradient accumulation index for tracking
        gradient_accumulation_idx = 0
        for mini_batch_start in range(0, args.local_batch_size, args.per_device_train_batch_size):
            mini_batch_end = mini_batch_start + args.per_device_train_batch_size

            # Prepare mini-batch data
            with accelerator.accumulate(policy):
                mb_query = data["llama_prompt_tokens"][mini_batch_start : mini_batch_end]

                # mb_chosen_response = data["llama_chosen_tokens"][mini_batch_start : mini_batch_end]
                # mb_chosen_reward = data["chosen_reward"][mini_batch_start : mini_batch_end]
                # mb_chosen_logprob = data["chosen_logprob"][mini_batch_start : mini_batch_end]

                # mb_reject_response = data["llama_reject_tokens"][mini_batch_start : mini_batch_end]
                # mb_reject_reward = data["reject_reward"][mini_batch_start : mini_batch_end]
                # mb_reject_logprob = data["reject_logprob"][mini_batch_start : mini_batch_end]

                mb_responses = [data[f"response_{i}_tokens"][mini_batch_start:mini_batch_end] for i in range(args.num_responses)]
                mb_rewards = [data[f"response_{i}_reward"][mini_batch_start:mini_batch_end] for i in range(args.num_responses)]

                mb_responses = torch.cat((mb_chosen_response, mb_reject_response), dim=0)
                mb_logprobs = torch.cat((mb_chosen_logprob, mb_reject_logprob), dim=0)
                mb_query_responses = torch.cat((torch.cat((mb_query, mb_query), dim=0), mb_responses), dim=1)
                mb_sequence_length = first_true_indices(mb_responses == tokenizer.pad_token_id) - 1
                mb_seq_mask = torch.arange(args.task.response_length, device=device).unsqueeze(0).expand_as(mb_responses) <= mb_sequence_length.unsqueeze(1)

                attention_mask = mb_query_responses != tokenizer.pad_token_id
                input_ids = torch.masked_fill(mb_query_responses, ~attention_mask, tokenizer.eos_token_id)

                output = policy(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True,
                )
                logits = output.logits[:, args.task.query_length - 1 : -1]
                logits /= args.task.temperature + 1e-7
                new_all_logprobs = F.log_softmax(logits, dim=-1)
                new_logprobs = torch.gather(new_all_logprobs, 2, input_ids[:, args.task.query_length:].unsqueeze(-1)).squeeze(-1)
                new_logprobs = (new_logprobs * mb_seq_mask).sum(-1)

                if update == 1:
                    print(('logprobs:', new_logprobs, mb_logprobs))
                # accelerator.print('5.5', new_logprobs[:args.per_device_train_batch_size], mb_logprobs[:args.per_device_train_batch_size])
                # accelerator.print('5.6', new_logprobs[args.per_device_train_batch_size:], mb_logprobs[args.per_device_train_batch_size:])
                # accelerator.print('6', mb_chosen_reward, mb_reject_reward)

                ratio_logprob = new_logprobs - mb_logprobs
                ratio_logprob = ratio_logprob[:args.per_device_train_batch_size] - ratio_logprob[args.per_device_train_batch_size:]
                reg_diff = ratio_logprob - args.rebel.eta * (mb_chosen_reward - mb_reject_reward)

                # accelerator.print(ratio_logprob, mb_chosen_reward - mb_reject_reward, reg_diff)

                loss = (reg_diff ** 2).mean()

                # accelerator.print(loss)

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    logprobs_diff = new_logprobs - mb_logprobs
                    ratio = torch.exp(logprobs_diff)
                    kl_stats[gradient_accumulation_idx] = logprobs_diff.mean()
                    chosen_kl_stats[gradient_accumulation_idx] = logprobs_diff[:args.per_device_train_batch_size].mean()
                    reject_kl_stats[gradient_accumulation_idx] = logprobs_diff[args.per_device_train_batch_size:].mean()
                    loss_stats[gradient_accumulation_idx] = loss
                    ratio_stats[gradient_accumulation_idx] = ratio.mean()
            gradient_accumulation_idx += 1
        if accelerator.is_main_process:
            console.print(
                f"update",
                update,
                "kl_stats",
                kl_stats.mean().item(),
                "loss",
                loss_stats.mean().item(),
            )

        with torch.no_grad():
            writer.add_scalar("objective/kl", accelerator.gather(kl_stats).mean().item(), update)
            writer.add_scalar("objective/chosen_kl", accelerator.gather(chosen_kl_stats).mean().item(), update)
            writer.add_scalar("objective/reject_kl", accelerator.gather(reject_kl_stats).mean().item(), update)
            writer.add_scalar("npg/loss/policy", accelerator.gather(loss).mean().item(), update)
            writer.add_scalar("npg/loss/policy_avg", accelerator.gather(loss_stats).mean().item(), update)
            
            writer.add_scalar("npg/val/ratio", accelerator.gather(ratio_stats).mean().item(), update)
            writer.add_scalar("npg/val/ratio_var", accelerator.gather(ratio_stats).var().item(), update)
            writer.add_scalar("npg/lr", lrnow, update)
            writer.add_scalar("npg/episode", global_step, update)
            eps = int(global_step / (time.time() - start_time))
            writer.add_scalar("npg/eps", eps, update)
            accelerator.print("npg/eps", eps, update)
            torch.cuda.empty_cache()

    # save model
    eval_dict = evaluate(args, accelerator.unwrap_model(policy), tokenizer, validation_dataloader)
    writer.add_scalar("objective/validation_loss", accelerator.gather(eval_dict["val_loss"]).mean().item(), update)
    writer.add_scalar("objective/validation_loss_eta=1", accelerator.gather(eval_dict["val_loss_eta=1"]).mean().item(), update)
    writer.add_scalar("objective/sign_align", accelerator.gather(eval_dict["sign_align"]).mean().item(), update)
    if args.output_dir:
        accelerator.wait_for_everyone()
        output_dir = os.path.join(args.output_dir, run_name) #!!!!! , str(update)
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        accelerator.save_state(output_dir=output_dir)
        accelerator.wait_for_everyone()
    torch.cuda.empty_cache()