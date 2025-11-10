import os
from datetime import datetime

from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from datasets import Dataset, load_dataset, concatenate_datasets

import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers

from grader_utils.parse_utils import parse_answer
from constants import *
from power_samp_utils import *


def extract_log_probs(output, generated_ids):
    """
    Extract log probabilities of chosen tokens from generate() output.

    Args:
        output: Output from model.generate() with output_scores=True
        generated_ids: The generated token IDs tensor

    Returns:
        tuple: (total_log_prob, avg_log_prob, per_token_log_probs)
    """
    if not hasattr(output, 'scores') or output.scores is None or len(output.scores) == 0:
        return 0.0, 0.0, []

    # Stack logits for all generated tokens
    # output.scores is a tuple of tensors with shape [batch_size, vocab_size]
    logits = torch.stack(output.scores, dim=0)  # [num_tokens, batch_size, vocab_size]

    # Squeeze out batch dimension if batch_size=1
    if logits.shape[1] == 1:
        logits = logits.squeeze(1)  # [num_tokens, vocab_size]

    # Convert to log probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # [num_tokens, vocab_size]

    # Ensure generated_ids is 1D
    if generated_ids.dim() > 1:
        generated_ids = generated_ids.squeeze()

    # Ensure we have the right length
    num_scores = len(output.scores)
    generated_ids = generated_ids[:num_scores]  # Trim to match scores length

    # Convert to CPU for safe indexing
    if generated_ids.is_cuda:
        generated_ids_cpu = generated_ids.cpu()
    else:
        generated_ids_cpu = generated_ids

    # Also move log_probs to CPU to avoid CUDA indexing issues
    log_probs_cpu = log_probs.cpu()

    # Validate token IDs are in valid range
    vocab_size = log_probs_cpu.shape[-1]
    if torch.any(generated_ids_cpu >= vocab_size) or torch.any(generated_ids_cpu < 0):
        print(f"WARNING: Invalid token IDs detected. Vocab size: {vocab_size}")
        print(f"Token ID range: [{generated_ids_cpu.min()}, {generated_ids_cpu.max()}]")
        # Clamp to valid range
        generated_ids_cpu = torch.clamp(generated_ids_cpu, 0, vocab_size - 1)

    # Convert to long tensor and ensure correct shape
    if not isinstance(generated_ids_cpu, torch.Tensor):
        generated_ids_cpu = torch.tensor(generated_ids_cpu, dtype=torch.long)
    generated_ids_cpu = generated_ids_cpu.long()

    # Get log probs of the chosen tokens using proper tensor indexing
    # Create index tensor for first dimension
    indices = torch.arange(num_scores, dtype=torch.long)
    chosen_log_probs = log_probs_cpu[indices, generated_ids_cpu]

    # Compute statistics
    total_log_prob = chosen_log_probs.sum().item()
    avg_log_prob = chosen_log_probs.mean().item()
    per_token_log_probs = chosen_log_probs.tolist()

    return total_log_prob, avg_log_prob, per_token_log_probs





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action = "store", type = str, default = "results/",  dest = "save_str")
    parser.add_argument("--model", action = "store", default = "qwen", type = str, choices = ["qwen", "qwen_math", "phi", "tulu", "qwen_math_grpo", "phi_grpo", "nemotron"])
    parser.add_argument("--temperature", action = "store", default = 0.25, type = float, dest = "temperature")
    parser.add_argument("--dataset", action = "store", default = "MATH", type = str, choices = ["MATH", "QuestA", "AIME"])
    parser.add_argument("--cot", action = "store", type = bool, default = True)
    parser.add_argument("--mcmc_steps", action = "store", type = int, default = 10)
    parser.add_argument("--device", action = "store", type = str, dest = "device", default = "cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--batch_idx", action = "store", type = int, default = 0)
    parser.add_argument("--seed", action = "store", type = int, default = 0)
    parser.add_argument("--block_num", action = "store", type = int, default = 16, help = "Number of blocks to divide generation into (progress bar blocks)")
    args = parser.parse_args()

    random.seed(0)


    model = args.model
    device = args.device
    dataset_name = args.dataset
    cot = args.cot
    temp = args.temperature
    mcmc_steps = args.mcmc_steps

    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)

    # Check for existing CSV files to resume from
    existing_results = []
    completed_questions = set()
    timestamp = None

    # Look for existing CSV files matching this batch/model/settings
    pattern = f"{model}_math_base_power_samp_batch{args.batch_idx}_mcmc{args.mcmc_steps}_B{args.block_num}_temp{args.temperature}_seed{args.seed}_*.csv"
    existing_files = glob(os.path.join(save_str, pattern))

    if existing_files:
        # Use the most recent file
        latest_file = max(existing_files, key=os.path.getmtime)
        print(f"\n{'='*60}")
        print(f"RESUMING from existing file:")
        print(f"{latest_file}")
        print(f"{'='*60}\n")

        try:
            existing_df = pd.read_csv(latest_file)
            existing_results = existing_df.to_dict('records')
            completed_questions = set(existing_df['question'].tolist())
            print(f"Found {len(existing_results)} completed questions")
            print(f"Will skip these and continue with remaining questions\n")

            # Extract timestamp from the existing filename (last 2 parts: YYYYMMDD_HHMMSS)
            filename_parts = os.path.basename(latest_file).replace('.csv', '').split('_')
            timestamp = '_'.join(filename_parts[-2:])
        except Exception as e:
            print(f"Warning: Could not load existing file: {e}")
            print("Starting fresh...\n")
            existing_results = []
            completed_questions = set()
            timestamp = None

    # Create timestamp for filename if not resuming
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Starting new run with timestamp: {timestamp}\n")

    print(model)
    print(device)
    print(mcmc_steps)
    if model == "qwen":
        model_str = "Qwen/Qwen2.5-7B"
    elif model == "qwen_math":
        model_str = "Qwen/Qwen2.5-Math-7B"
    elif model == "qwen_math_grpo":
        model_str = "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150"
    elif model == "phi":
        model_str = 'microsoft/Phi-3.5-mini-instruct'
    elif model == "tulu":
        model_str = "allenai/Llama-3.1-Tulu-3-8B-DPO"
    elif model == "nemotron":
        model_str = "nvidia/OpenMath-Nemotron-1.5B"

    if dataset_name == "MATH":
        json_file = 'data/MATH500.json'
        dataset = json.load(open(json_file, "r"))
    elif dataset_name == "QuestA":
        from datasets import load_dataset
        hf_dataset = load_dataset("foreverlasting1202/QuestA", split="train")
        # Convert to list of dicts and rename 'problem' to 'prompt' for compatibility
        dataset = [{"prompt": item["problem"], "answer": item["answer"]} for item in hf_dataset]
    elif dataset_name == "AIME":
        from datasets import load_dataset
        hf_dataset = load_dataset("gneubig/aime-1983-2024", split="train")
        # Filter to only keep data up to 2023 and convert to expected format
        dataset = [{"prompt": item["Question"], "answer": item["Answer"]}
                   for item in hf_dataset if item["Year"] <= 2023]



    print("dataset done")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code = True)

    # Set pad_token to eos_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_str, dtype="auto", device_map="auto", trust_remote_code = True)
    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)


    print("loaded models")
    results = existing_results  # Start with existing results if resuming

    start = 100*args.batch_idx
    end = 100*(args.batch_idx+1)

    for problem, data in tqdm(enumerate(dataset[start:end]), desc = "Benchmark on MATH"):
        question = data["prompt"]

        # Skip if this question was already completed
        if question in completed_questions:
            print(f"\n[SKIPPING] Question already completed: {question[:80]}...\n")
            continue

        print(question)
        answer = data["answer"]

        input_text = format_prompt(question, model, tokenizer, cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        prefx = [idx.item() for idx in input_ids[0]]

        # Naive generation
        naive_start_time = time.time()
        attention_mask = torch.ones_like(input_ids)
        naive_temp_output = hf_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=16000,
                                return_dict_in_generate=True, output_scores=True, temperature=temp,
                                do_sample=True, pad_token_id=tokenizer.pad_token_id)
        naive_elapsed_time = time.time() - naive_start_time

        naive_generated_ids = naive_temp_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        naive_completion = tokenizer.decode(naive_generated_ids, skip_special_tokens=True)
        naive_total_logprob, naive_avg_logprob, _ = extract_log_probs(naive_temp_output, naive_generated_ids)
        naive_num_tokens = len(naive_generated_ids) if naive_generated_ids.dim() == 1 else naive_generated_ids.shape[0]

        # Delete output tensors to free GPU memory
        del naive_temp_output
        torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print("NAIVE (low temp) DONE:")
        print(f"Time: {naive_elapsed_time:.2f}s | Tokens: {naive_num_tokens} | Total log prob: {naive_total_logprob:.2f} | Avg log prob: {naive_avg_logprob:.4f}")
        print(f"{'='*60}")
        print(naive_completion)
        print(f"{'='*60}\n")


        # Standard generation
        std_start_time = time.time()
        std_output = hf_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=16000,
                                return_dict_in_generate=True, output_scores=True, do_sample = True,
                                pad_token_id=tokenizer.pad_token_id)
        std_elapsed_time = time.time() - std_start_time

        std_generated_ids = std_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        std_completion = tokenizer.decode(std_generated_ids, skip_special_tokens=True)
        std_total_logprob, std_avg_logprob, _ = extract_log_probs(std_output, std_generated_ids)
        std_num_tokens = len(std_generated_ids) if std_generated_ids.dim() == 1 else std_generated_ids.shape[0]

        # Delete output tensors to free GPU memory
        del std_output
        torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print("STD (temp=1.0) DONE:")
        print(f"Time: {std_elapsed_time:.2f}s | Tokens: {std_num_tokens} | Total log prob: {std_total_logprob:.2f} | Avg log prob: {std_avg_logprob:.4f}")
        print(f"{'='*60}")
        print(std_completion)
        print(f"{'='*60}\n")


        # MCMC generation
        mcmc_start_time = time.time()
        mcmc_power_samp_output, log_probs_norm, log_probs_unnorm, acceptance_ratio = mcmc_power_samp(autoreg_sampler, prefx, temp, mcmc_steps, max_new_tokens=16000, block_num=args.block_num)
        mcmc_elapsed_time = time.time() - mcmc_start_time

        mcmc_power_samp_ids = torch.tensor([mcmc_power_samp_output], dtype=torch.long, device=device).squeeze().to("cpu")
        mcmc_completion = tokenizer.decode(mcmc_power_samp_ids, skip_special_tokens=True)
        mcmc_total_logprob = sum(log_probs_unnorm) if log_probs_unnorm else 0.0
        mcmc_avg_logprob = (sum(log_probs_unnorm) / len(log_probs_unnorm)) if log_probs_unnorm else 0.0
        mcmc_num_tokens = len(mcmc_power_samp_ids) if mcmc_power_samp_ids.dim() == 1 else mcmc_power_samp_ids.shape[0]

        print(f"\n{'='*60}")
        print("MCMC DONE:")
        print(f"Time: {mcmc_elapsed_time:.2f}s | Tokens: {mcmc_num_tokens} | Total log prob: {mcmc_total_logprob:.2f} | Avg log prob: {mcmc_avg_logprob:.4f}")
        print(f"Acceptance ratio: {acceptance_ratio:.4f}")
        print(f"{'='*60}")
        print(mcmc_completion)
        print(f"{'='*60}\n")

        naive_answer = parse_answer(naive_completion)
        std_answer = parse_answer(std_completion)
        mcmc_answer = parse_answer(mcmc_completion)

        print(f"\n{'='*60}")
        print("ANSWERS:")
        print(f"Correct answer: {answer}")
        print(f"Naive answer:   {naive_answer}")
        print(f"Std answer:     {std_answer}")
        print(f"MCMC answer:    {mcmc_answer}")
        print(f"{'='*60}\n")


        results.append({
            "question": question,
            "correct_answer": answer,
            "naive_completion": naive_completion,
            "naive_answer": naive_answer,
            "naive_time": naive_elapsed_time,
            "naive_num_tokens": naive_num_tokens,
            "naive_total_logprob": naive_total_logprob,
            "naive_avg_logprob": naive_avg_logprob,
            "std_completion": std_completion,
            "std_answer": std_answer,
            "std_time": std_elapsed_time,
            "std_num_tokens": std_num_tokens,
            "std_total_logprob": std_total_logprob,
            "std_avg_logprob": std_avg_logprob,
            "mcmc_completion": mcmc_completion,
            "mcmc_answer": mcmc_answer,
            "mcmc_time": mcmc_elapsed_time,
            "mcmc_num_tokens": mcmc_num_tokens,
            "mcmc_total_logprob": mcmc_total_logprob,
            "mcmc_avg_logprob": mcmc_avg_logprob,
            "acceptance_ratio": acceptance_ratio,
        })

        # Save after each problem to avoid losing progress
        df = pd.DataFrame(results)
        filename = f"{model}_math_base_power_samp_batch{args.batch_idx}_mcmc{mcmc_steps}_B{args.block_num}_temp{temp}_seed{args.seed}_{timestamp}.csv"
        df.to_csv(os.path.join(save_str, filename), index=False)

        # Clear GPU cache after each problem to prevent memory accumulation
        torch.cuda.empty_cache()


    # Final save (redundant but kept for safety)
    df = pd.DataFrame(results)
    filename = f"{model}_math_base_power_samp_batch{args.batch_idx}_mcmc{mcmc_steps}_B{args.block_num}_temp{temp}_seed{args.seed}_{timestamp}.csv"
    full_path = os.path.join(save_str, filename)
    df.to_csv(full_path, index=False)

    # Print final summary
    print(f"\n{'='*60}")
    print(f"COMPLETED!")
    print(f"Total questions in results: {len(results)}")
    print(f"Questions completed this run: {len(results) - len(existing_results)}")
    print(f"Saved to: {full_path}")
    print(f"{'='*60}\n")













        













