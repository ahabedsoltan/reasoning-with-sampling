import os
from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from grader_utils.parse_utils import parse_answer
from constants import *

from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessor

from raw_logits_tap import RawLogitsTap
import vllm.v1.sample.sampler as v1_sampler

import torch
import torch.nn.functional as F

import ipdb


def chosen_token_logp_from_topk(raw_logits_list, tokens, temp=1.0):
    log_probs_norm = []
    log_probs_unnorm = []

    for step_dict, tok in zip(raw_logits_list, tokens):
        # sort by rank
        items = sorted(step_dict.items(), key=lambda x: x[1].rank)

        step_token_ids = torch.tensor([tid for tid, _ in items])
        step_logits = torch.tensor([info.logprob for _, info in items])

        # temperature scaling
        scaled_logits = step_logits / temp

        # normalized log-probs over top-k only
        log_probs_scaled = F.log_softmax(scaled_logits, dim=-1)
        log_probs_unscaled = (1 / temp) * F.log_softmax(step_logits, dim=-1)

        # chosen token
        pos = (step_token_ids == tok).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            log_probs_norm.append(float("nan"))
            log_probs_unnorm.append(float("nan"))
        else:
            log_probs_norm.append(log_probs_scaled[pos[0]].item())
            log_probs_unnorm.append(log_probs_unscaled[pos[0]].item())

    return log_probs_unnorm, log_probs_norm


### DESCRIPTION ###
# vLLM-based power sampling to sample from p^{alpha}, where p is the base model
# takes in 1/alpha (temperature) as an argument (default 0.25), and mcmc_power_samp implements sampling from p^{alpha}


class VLLMAutogregressiveSampler:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # vLLM handles device management internally


# returns probabilities (normed)
def normalize(dist):
    probs = F.softmax(dist, dim=-1)
    return probs


# returns sum of logits (product of distributions p*q)
def dist_product(logit_p, logit_q):
    return logit_p + logit_q


# returns logit scaled by temp (temperature scaling p^(1/tau))
def dist_temp_scale(logit_p, temp):
    return logit_p * torch.tensor(1 / temp, dtype=logit_p.dtype)


# low-temperature sampling proposal distribution using vLLM
def naive_temp_vllm(p: VLLMAutogregressiveSampler, context, temp, seq_len, number_log_probs_to_use=1000):
    c = len(context)
    tokenizer = p.tokenizer

    # Decode context to text
    context_text = tokenizer.decode(context)

    # Set up sampling parameters for vLLM
    sampling_params = SamplingParams(
        temperature=temp,
        max_tokens=seq_len - c,
        logprobs=number_log_probs_to_use,
    )

    outputs = p.model.generate([context_text], sampling_params, use_tqdm=False)
    output = outputs[0]

    # Extract generated tokens using vLLM's actual token IDs
    tokens = output.outputs[0].token_ids
    prop = context + tokens
    raw_logits_list = output.outputs[0].logprobs

    log_probs_unnorm, log_probs_norm = chosen_token_logp_from_topk(
        raw_logits_list, tokens, temp=temp
    )

    return prop, log_probs_norm, log_probs_unnorm


# power sampling with autoregressive mcmc using vLLM
def mcmc_power_samp_vllm(
    p: VLLMAutogregressiveSampler,
    context,
    temp,
    mcmc_steps,
    max_new_tokens,
    number_log_probs_to_use=1000,
    block_num=4,
):
    c = len(context)

    # basic checks & derived quantities
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)

    # nice config print
    print("\n┌── MCMC POWER SAMPLING CONFIG ─────────────────────────────")
    print(f"│ alpha (1/temp)      : {1 / temp:.4f}")
    print(f"│ temperature (temp)  : {temp:.4f}")
    print(f"│ max_new_tokens      : {max_new_tokens}")
    print(f"│ block_num           : {block_num}")
    print(f"│ jump_size           : {jump_size}")
    print("└" + "─" * 58)

    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []

    attempts = 0
    acceptances = 0

    for blocki in tqdm(range(block_num), desc="MCMC blocks"):
        # proposal from low-T model on current prefix
        gen, lp_norm, lp_unnorm = naive_temp_vllm(
            p,
            gen,
            number_log_probs_to_use=number_log_probs_to_use,
            temp=temp,
            seq_len=jump_size + len(gen),
        )
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        cum_lp = sum(log_probs_norm)
        print("\n┌── BLOCK {}/{} SUMMARY ─────────────────────────────".format(blocki + 1, block_num))
        print(f"│ cumulative log_prob_norm sum : {cum_lp:.3f}")
        print("└" + "─" * 58)

        for stepi in tqdm(range(mcmc_steps), desc=f"MCMC steps (block {blocki+1})", leave=False):
            attempts += 1
            t = len(gen)
            idx = random.randint(c, t - 1)

            # proposal by resampling suffix from prefix up to idx
            prop, log_prob_prop, target_log_prob_prop = naive_temp_vllm(
                p, gen[:idx], temp=temp, seq_len=t, number_log_probs_to_use=number_log_probs_to_use
            )

            print(
                f"[block {blocki+1}/{block_num} | step {stepi+1}/{mcmc_steps}] "
                f"cumulative_log_prob_norm={sum(log_probs_norm):.3f}"
            )

            s = len(prop)
            assert len(log_prob_prop) == s - idx
            assert len(target_log_prob_prop) == s - idx

            log_prob_cur = log_probs_norm.copy()[idx - c : s - c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx - c : s - c]

            log_r = (
                sum(target_log_prob_prop)
                + sum(log_prob_cur)
                - sum(target_log_prob_cur)
                - sum(log_prob_prop)
            )

            if np.random.rand() < np.exp(log_r):
                acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx - c :] = log_prob_prop.copy()
                log_probs_unnorm[idx - c :] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        generated_part = gen[c:]  # Only look at sampled tokens

        if p.tokenizer.eos_token_id in generated_part:
            local_eos = generated_part.index(p.tokenizer.eos_token_id)
            eos_idx = c + local_eos  # shift back to global index

            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx - c + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx - c + 1]

            acceptance_ratio = acceptances / attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

        # # early stop if EOS reached
        # if p.tokenizer.eos_token_id in gen:
        #     eos_idx = gen.index(p.tokenizer.eos_token_id)
        #     gen = gen[: eos_idx + 1]
        #     log_probs_norm = log_probs_norm[: eos_idx + 1]
        #     log_probs_unnorm = log_probs_unnorm[: eos_idx + 1]
        #     acceptance_ratio = acceptances / attempts
        #     return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances / attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio


def format_prompt(question, model, tokenizer, cot=True):
    if model == "qwen":
        format_str = PROMPT + question
        if cot:
            format_str += COT
        else:
            format_str += BASE

    elif model == "qwen_math":
        format_str = PROMPT + question
        if cot:
            format_str += COT
        else:
            format_str += BASE

    elif model == "qwen_math_grpo":
        content_str = PROMPT + question
        if cot:
            content_str += COT
        else:
            content_str += BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    elif model == "phi_grpo":
        content_str = PROMPT + question
        if cot:
            content_str += COT
        else:
            content_str += BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    elif model == "phi":
        content_str = PROMPT + question
        if cot:
            content_str += COT
        else:
            content_str += BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    elif model == "tulu":
        content_str = PROMPT + question
        if cot:
            content_str += COT
        else:
            content_str += BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    elif model == "nemotron":
        content_str = PROMPT + question
        if cot:
            content_str += COT
        else:
            content_str += BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context, tokenize=False, add_generation_prompt=True
        )

    return format_str
