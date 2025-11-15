from datetime import datetime

from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse

import os
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

import sys
import importlib

from power_samp_utils_vllm import chosen_token_logp_from_topk
from vllm import LLM, SamplingParams

from grader_utils.parse_utils import parse_answer
from constants import *
from power_samp_utils_vllm import *

import ipdb


# -------------------------------------------------------
# PRETTY CONFIG HEADER
# -------------------------------------------------------
def pretty_print_config(args):
    line = "=" * 80
    print("\n" + line)
    print("EXPERIMENT CONFIGURATION")
    print(line)

    print(f"  --model                   = {args.model}")
    print(f"  --dataset                 = {args.dataset}")
    print(f"  --mcmc_steps              = {args.mcmc_steps}")
    print(f"  --temperature             = {args.temperature}")
    print(f"  --seed                    = {args.seed}")
    print(f"  --number_log_probs_to_use = {args.number_log_probs_to_use}")
    print(f"  --mcmc_block_size         = {args.mcmc_block_size}")
    print(f"  --max_context_length      = {args.max_context_length}")

    print(f"  --tensor_parallel_size    = {args.tensor_parallel_size}")
    print(f"  --gpu_memory_utilization  = {args.gpu_memory_utilization}")
    print(f"  --batch_idx               = {args.batch_idx}")
    print(f"  --cot                     = {args.cot}")
    print(f"  --save_str                = {args.save_str}")

    # NEW: which modes are running
    print(f"  --run_naive               = {args.run_naive}")
    print(f"  --run_std                 = {args.run_std}")
    print(f"  --run_mcmc                = {args.run_mcmc}")

    print(line)
    print(
        "CLI FLAGS: "
        f"--model={args.model} "
        f"--dataset={args.dataset} "
        f"--mcmc_steps={args.mcmc_steps} "
        f"--temperature={args.temperature} "
        f"--seed={args.seed} "
        f"--number_log_probs_to_use={args.number_log_probs_to_use} "
        f"--mcmc_block_size={args.mcmc_block_size} "
        f"--max_context_length={args.max_context_length} "
        f"--run_naive={args.run_naive} "
        f"--run_std={args.run_std} "
        f"--run_mcmc={args.run_mcmc} "
    )
    print(line + "\n")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_str", action="store", type=str, default="results/", dest="save_str")
    parser.add_argument("--model", action="store", default="qwen", type=str,
                        choices=["qwen", "qwen_math", "phi", "tulu", "qwen_math_grpo", "phi_grpo", "nemotron"])
    parser.add_argument("--temperature", action="store", default=0.25, type=float, dest="temperature")
    parser.add_argument("--dataset", action="store", default="MATH", type=str,
                        choices=["MATH", "QuestA", "AIME"])
    parser.add_argument("--cot", action="store", type=bool, default=True)
    parser.add_argument("--mcmc_steps", action="store", type=int, default=10)
    parser.add_argument("--batch_idx", action="store", type=int, default=0)
    parser.add_argument("--seed", action="store", type=int, default=0)
    parser.add_argument("--tensor_parallel_size", action="store", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", action="store", type=float, default=0.9)
    parser.add_argument("--number_log_probs_to_use", action="store", type=int, default=1000)
    parser.add_argument("--max_context_length", action="store", type=int, default=4000)
    parser.add_argument("--mcmc_block_size", action="store", type=int, default=2)

    # NEW: flags to control which methods to run (0/1)
    parser.add_argument("--run_naive", type=int, default=1, choices=[0, 1])
    parser.add_argument("--run_std", type=int, default=1, choices=[0, 1])
    parser.add_argument("--run_mcmc", type=int, default=1, choices=[0, 1])

    args = parser.parse_args()
    pretty_print_config(args)

    random.seed(args.seed)

    model = args.model
    dataset_name = args.dataset
    cot = args.cot
    temp = args.temperature

    save_str = os.path.join(args.save_str, model + "_vllm")
    os.makedirs(save_str, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Model key       : {model}")
    print(f"Dataset         : {dataset_name}")
    print(f"Using vLLM with tensor_parallel_size={args.tensor_parallel_size}")
    print(f"MCMC steps      : {args.mcmc_steps}")

    # Model mapping
    if model == "qwen":
        model_str = "Qwen/Qwen2.5-7B"
    elif model == "qwen_math":
        model_str = "Qwen/Qwen2.5-Math-7B"
    elif model == "qwen_math_grpo":
        model_str = "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150"
    elif model == "phi":
        model_str = "microsoft/Phi-3.5-mini-instruct"
    elif model == "tulu":
        model_str = "allenai/Llama-3.1-Tulu-3-8B-DPO"
    elif model == "nemotron":
        model_str = "nvidia/OpenMath-Nemotron-1.5B"
    else:
        raise ValueError("Unknown model")

    # Dataset loading
    if dataset_name == "MATH":
        dataset = json.load(open("/u/abedsol1/research/reason/llm_experiments/data/MATH500.json"))
    elif dataset_name == "QuestA":
        hf = load_dataset("foreverlasting1202/QuestA", split="train")
        dataset = [{"prompt": x["problem"], "answer": x["answer"]} for x in hf]
    elif dataset_name == "AIME":
        hf = load_dataset("gneubig/aime-1983-2024", split="train")
        dataset = [{"prompt": x["Question"], "answer": x["Answer"]} for x in hf if x["Year"] <= 2023]

    print("Dataset loaded.")

    # Load vLLM
    vllm_model = LLM(
        model=model_str,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_logprobs=args.number_log_probs_to_use,
        dtype="auto",
        logprobs_mode="raw_logits",
        max_model_len=args.max_context_length,
    )
    tokenizer = vllm_model.get_tokenizer()
    vllm_sampler = VLLMAutogregressiveSampler(vllm_model, tokenizer)

    print("Models loaded.")

    results = []
    start = 100 * args.batch_idx
    end = 100 * (args.batch_idx + 1)

    # -------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------
    for idx_global, (idx_local, data) in enumerate(
        tqdm(enumerate(dataset[start:end]), desc="Benchmark on MATH"), start=start
    ):
        question = data["prompt"]
        answer = data["answer"]

        print("\n" + "-" * 80)
        print(f"[Problem #{idx_global}]")
        print("-" * 80)
        print("QUESTION:")
        print(question)
        print("-" * 80)

        input_text = format_prompt(question, model, tokenizer, cot)
        input_ids = tokenizer.encode(input_text)
        prefix = input_ids

        # -------------------------------------------------------
        # NAIVE
        # -------------------------------------------------------
        if args.run_naive:
            naive_start = time.time()
            naive_params = SamplingParams(
                temperature=temp,
                max_tokens=args.max_context_length,
                logprobs=args.number_log_probs_to_use,
            )
            naive_output = vllm_model.generate([input_text], naive_params, use_tqdm=False)[0]
            naive_time = time.time() - naive_start

            naive_text = naive_output.outputs[0].text
            naive_tokens = naive_output.outputs[0].token_ids
            naive_logits = naive_output.outputs[0].logprobs
            lp_unnorm_naive, lp_norm_naive = chosen_token_logp_from_topk(
                naive_logits, naive_tokens, temp
            )

            print("┌── NAIVE " + "─" * 40)
            print(f"│ time          : {naive_time:.2f} s")
            print(f"│ log_prob_sum  : {sum(lp_norm_naive):.3f}")
            print(f"│ #tokens       : {len(naive_tokens)}")
            print("└" + "─" * 48)
            print(naive_text)
        else:
            naive_time = 0.0
            naive_text = ""
            naive_tokens = []
            lp_norm_naive = []

        # -------------------------------------------------------
        # STANDARD 1.0
        # -------------------------------------------------------
        if args.run_std:
            std_start = time.time()
            std_params = SamplingParams(
                temperature=1.0,
                max_tokens=args.max_context_length,
                logprobs=args.number_log_probs_to_use,
            )
            std_output = vllm_model.generate([input_text], std_params, use_tqdm=False)[0]
            std_time = time.time() - std_start

            std_text = std_output.outputs[0].text
            std_tokens = std_output.outputs[0].token_ids
            std_logits = std_output.outputs[0].logprobs
            _, lp_norm_std = chosen_token_logp_from_topk(
                std_logits, std_tokens, temp=1.0
            )

            print("┌── STD (T=1.0) " + "─" * 34)
            print(f"│ time          : {std_time:.2f} s")
            print(f"│ log_prob_sum  : {sum(lp_norm_std):.3f}")
            print("└" + "─" * 48)
            print(std_text)
        else:
            std_time = 0.0
            std_text = ""
            std_tokens = []
            lp_norm_std = []

        # -------------------------------------------------------
        # MCMC
        # -------------------------------------------------------
        if args.run_mcmc:
            mcmc_start = time.time()
            mcmc_out, lp_norm_mcmc, _, acc = mcmc_power_samp_vllm(
                vllm_sampler,
                prefix,
                temp,
                args.mcmc_steps,
                number_log_probs_to_use=args.number_log_probs_to_use,
                max_new_tokens=args.max_context_length,
                block_num=args.mcmc_block_size,
            )
            mcmc_time = time.time() - mcmc_start

            mcmc_text = tokenizer.decode(mcmc_out[len(prefix):], skip_special_tokens=True)

            print("┌── MCMC " + "─" * 41)
            print(f"│ time          : {mcmc_time:.2f} s")
            print(f"│ log_prob_sum  : {sum(lp_norm_mcmc):.3f}")
            print(f"│ #tokens       : {len(mcmc_out) - len(prefix)}")
            print(f"│ acceptance    : {acc:.3f}")
            print("└" + "─" * 48)
            print(mcmc_text)
        else:
            mcmc_time = 0.0
            mcmc_text = ""
            lp_norm_mcmc = []
            acc = 0.0

        # -------------------------------------------------------
        # Parsed answers
        # -------------------------------------------------------
        naive_answer = parse_answer(naive_text) if args.run_naive else None
        std_answer = parse_answer(std_text) if args.run_std else None
        mcmc_answer = parse_answer(mcmc_text) if args.run_mcmc else None

        print("\nPARSED ANSWERS:")
        print(f"  correct : {answer}")
        if args.run_naive:
            print(f"  naive   : {naive_answer} -- log_prob_sum  : {sum(lp_norm_naive):.3f}")
        else:
            print("  naive   : [SKIPPED]")
        if args.run_std:
            print(f"  std     : {std_answer} -- log_prob_sum  : {sum(lp_norm_std):.3f}")
        else:
            print("  std     : [SKIPPED]")
        if args.run_mcmc:
            print(f"  mcmc    : {mcmc_answer} -- log_prob_sum:{sum(lp_norm_mcmc):.3f}")
            print(f"  acceptance ratio: {acc:.3f}")
        else:
            print("  mcmc    : [SKIPPED]")

        results.append({
            "question": question,
            "correct_answer": answer,

            "naive_completion": naive_text,
            "naive_answer": naive_answer,
            "naive_time": naive_time,

            "std_completion": std_text,
            "std_answer": std_answer,
            "std_time": std_time,

            "mcmc_completion": mcmc_text,
            "mcmc_answer": mcmc_answer,
            "mcmc_time": mcmc_time,
            "acceptance_ratio": acc,
        })

        df = pd.DataFrame(results)
        fname = f"{model}_math_vllm_power_samp_batch{args.batch_idx}_mcmc{args.mcmc_steps}_temp{temp}_seed{args.seed}_{timestamp}.csv"
        df.to_csv(os.path.join(save_str, fname), index=False)

    # Final save
    df = pd.DataFrame(results)
    fname = f"{model}_math_vllm_power_samp_batch{args.batch_idx}_mcmc{args.mcmc_steps}_temp{temp}_seed{args.seed}_{timestamp}.csv"
    df.to_csv(os.path.join(save_str, fname), index=False)
