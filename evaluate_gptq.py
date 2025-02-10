import sys
import argparse
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from learn_to_quant import eval_utils
import wandb
import json
import os
from collections import defaultdict 
import pdb 
import gc
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig


from lm_eval.models import huggingface
from lm_eval import simple_evaluate
import torch 

import datasets

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE=True

def evaluate_with_harness_full(model, tokenizer, device, debug=False, batch_size=2):
    """
    Evaluates a causall LLM model using evaluation harness on the full dataset, unlike def evaluate_with_harness, 
    which is only on a small susbet 

    Args:
        model (hf model )
        device (str, optional): The device to use for the evaluation ('cpu' or 'cuda'). Default is 'cpu'.
        debug (bool, optional): Whether to run the evaluation in debug mode or not. Default is False.
        batch_size (int, optional): The batch size to use for the evaluation. Default is 2.

    Returns:
        dict: A dictionary containing the evaluation metrics, including the accuracy on the MMLU (MultiModal Lexical Understanding) social sciences task and the exact match accuracy on the Natural Questions (NQ) open-ended task.
    """
    import time
    

    start = time.time()
    model = model.eval() 
    lm_obj = huggingface.HFLM(pretrained=model, backend='causal', tokenizer=tokenizer, batch_size=batch_size, device=device)

    if debug: 
       limit1 = 2
    else: 
       limit1 = None
       
    all_metrics = {}

    results1 = simple_evaluate( # call simple_evaluate
            model=lm_obj,
            tasks=["mmlu", "boolq"],
            num_fewshot=0,
            limit=limit1,
            batch_size=batch_size,
            cache_requests=None,
            log_samples=False,
            bootstrap_iters=2,
            gen_kwargs="max_new_tokens=40",
        )
    
    results_qa = simple_evaluate( # call simple_evaluate
        model=lm_obj,
        tasks=['nq_open'],
        num_fewshot=5,
        limit=limit1,
        device = 'cuda',
        batch_size=batch_size,
        cache_requests=None,
        log_samples=False,
        gen_kwargs="max_new_tokens=40",
        bootstrap_iters=2
    )

    all_metrics = {f'eval_harness_shot=0/{key}': results1['results'][key]['acc,none'] for key in results1['results'] if 'mmlu' not in key}
    all_metrics[f'eval_harness_shot=0/mmlu'] = results1['results']['mmlu']['acc,none']

    tmp = {f'eval_harness_shot=5/{key}': results_qa['results'][key]['exact_match,remove_whitespace'] for key in results_qa['results']}
    all_metrics.update(tmp)

    print(f'Completed evaluation with harness in {time.time()-start: 0.3f} seconds')
    return all_metrics

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quantization Error Analysis')

    parser.add_argument('--model_name', type=str, required=False, help='ID of the base model', default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--cache_dir', type=str, required=False, help='Cache directory for loading HF models', default=None)
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--exp_name', type=str, default='eval_gptq', help='Experiment name for logging')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--bits', type=int, default=4, help='Number of quantization bits', choices=[2,3,4,8])

    args = parser.parse_args()

    # Set up Weights and Biases logging
    run = wandb.init(project="learn-to-quantize", name=args.exp_name, dir=args.cache_dir)

    calibration_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train"
    ).select(range(1024))["text"]

    if args.debug: 
        calibration_dataset = [" ".join(item.split()[:30]) for item in calibration_dataset] # speedup

    quantize_config = QuantizeConfig(
        bits=args.bits, # works with bit=4
        group_size=64,
    )

    model = GPTQModel.load(args.model_name, quantize_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    # increase `batch_size` to match gpu/vram specs to speed up quantization
    model.quantize(calibration_dataset, batch_size=args.batch_size)

    all_metrics = {} 

    # Evaluate with harnes
    print(f'Evaluating model on device: {model.device}')
    harness_metrics = evaluate_with_harness_full(model, tokenizer, model.device, debug=args.debug, batch_size=args.batch_size)
    all_metrics['eval_metrics'] = harness_metrics

    # Finish Weights and Biases run
    wandb.finish()