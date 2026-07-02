#!/usr/bin/env python
# coding: utf-8

import dct as dct
from tqdm import tqdm
import math
from torch import vmap
import torch
import os
import argparse
from datasets import load_dataset, load_from_disk
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
torch.backends.cuda.enable_mem_efficient_sdp(False)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DCT models with multiple scaling factors')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen1.5-32B-Chat",
                        help='Model name to use')
    parser.add_argument('--tokenizer_name', type=str, default=None,
                        help='Tokenizer name to use (defaults to model_name if not specified)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='HuggingFace dataset to use for instructions')
    parser.add_argument('--split', type=str, default=None,
                        help='Dataset split to use')
    parser.add_argument('--field', type=str, default="prompt",
                        help='Field from dataset to use as instructions')
    parser.add_argument('--num_samples', type=int, default=32,
                        help='Number of samples to use from dataset')
    parser.add_argument('--system_prompt', type=str, 
                        default="You are a helpful and harmless assistant.",
                        help='System prompt to use')
    parser.add_argument('--ortho', action=argparse.BooleanOptionalAction,
                        help='Enable orthogonalization')
    parser.add_argument('--deflate', action=argparse.BooleanOptionalAction,
                        help='Enable simultaneous deflation')
    parser.add_argument('--separate_u', action=argparse.BooleanOptionalAction,
                        help='Let output directions vary across data-points')
    parser.add_argument('--deflation_iterations', type=int, default=10,
                        help='Number deflation iterations')
    parser.add_argument('--deflation_temp', type=float, default=1.0,
                        help='Deflation temperature')
    parser.add_argument('--source_layer_idx', type=int, default=10,
                        help='Source layer index')
    parser.add_argument('--target_layer_idx', type=int, default=20,
                        help='Target layer index')
    parser.add_argument('--num_factors', type=int, default=1024,
                        help='Number of factors to learn')
    parser.add_argument('--num_iters', type=int, default=30,
                        help='Number of iterations')
    parser.add_argument('--token_idxs', type=str, default="-3:",
                        help='Target token positions (Python slice notation)')
    parser.add_argument('--forward_batch_size', type=int, default=1,
                        help='Batch size for forward passes')
    parser.add_argument('--backward_batch_size', type=int, default=1,
                        help='Batch size for backward passes')
    parser.add_argument('--factor_batch_size', type=int, default=16,
                        help='Factor batch size')
    parser.add_argument('--scale', type=float, default=None,
                        help='Scale parameter. If missing will perform calibration')
    parser.add_argument('--scalar_multipliers', type=str, default="1.0",
                        help='Comma-separated list of multipliers for input_scale')
    parser.add_argument('--output_dir', type=str, default="./outputs",
                        help='Directory to save outputs')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Device to run on (cuda, cpu)')
    parser.add_argument('--calibration_sample_size', type=int, default=30,
                        help='Sample size for random directions used for calibration')
    parser.add_argument('--calibration_prompt_sample_size', type=int, default=1,
                        help='Prompt sample size for calibrating input scale')
    parser.add_argument('--target_ratio', type=float, default=0.5,
                        help='Target ratio for calibration')
    parser.add_argument('--save_normalized', action='store_true',
                        help='Save normalized V matrices instead of scaling them')
    parser.add_argument('--output_prefix', type=str, default="dct_vectors",
                        help='Prefix for output files')
    parser.add_argument('--for_each', action='store_true',
                        help='Train separate DCT model for each observation')
    parser.add_argument('--data_start', type=int,
                        help='Start observation to use in dataset')
    parser.add_argument('--data_end', type=int,
                        help='End observation to use in dataset')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Max sequence length beyond which training sequences will be truncated')

    return parser.parse_args()

def save_objective_plots(args, scalar_multipliers, exp_dct, multiplier, output_dir, output_prefix):
    """Save a plot of the objective values during training."""
    plt.figure(figsize=(10, 6))
    iterations = np.arange(1, len(exp_dct.objective_values) + 1)
    plt.plot(iterations, exp_dct.objective_values, marker='o', linestyle='-', markersize=4)
    plt.title(f'DCT Objective Values (Scale Multiplier: {multiplier})')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.grid(True, alpha=0.3)
    
    # Add annotations for key parameters
    param_text = (
        f"Model: {args.model_name.split('/')[-1]}\n"
        f"Source Layer: {args.source_layer_idx}\n"
        f"Target Layer: {args.target_layer_idx}\n"
        f"Num Factors: {args.num_factors}\n"
        f"Input Scale: {multiplier}"
    )
    plt.annotate(param_text, xy=(0.02, 0.97), xycoords='axes fraction', 
                 va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    
    # Save the figure
    plt_path = os.path.join(output_dir, f"{output_prefix}_objective_plot_scale_{multiplier}.png")
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved objective plot to {plt_path}")

def plot_gram_matrix_density(V, output_dir, output_prefix, multiplier):
    """
    Plot the density of the absolute values of the off-diagonal elements of the gram matrix of V.
    
    Args:
        V: Tensor of shape (d, n) where d is the dimension and n is the number of vectors
        output_dir: Directory to save the plot
        output_prefix: Prefix for the output file name
        multiplier: Current scale multiplier (for filename)
    """
    # Compute the Gram matrix
    gram = V.t() @ V
    
    # Get the off-diagonal elements
    mask = ~torch.eye(gram.shape[0], dtype=torch.bool, device=V.device)
    off_diag = gram[mask]
    
    # Convert to absolute values and move to CPU for plotting
    off_diag_abs = off_diag.abs().cpu().numpy()
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Use seaborn's displot for the density plot
    ax = sns.kdeplot(
        off_diag_abs, 
        fill=True, 
        cut=0
    )
    
    plt.title(f'Density of |Off-Diagonal Elements| in V Gram Matrix (Scale Multiplier: {multiplier})')
    plt.xlabel('Absolute Value of Off-Diagonal Elements')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # Add statistics annotation
    stats_text = (
        f"Mean: {off_diag_abs.mean():.4f}\n"
        f"Median: {np.median(off_diag_abs):.4f}\n"
        f"Std Dev: {off_diag_abs.std():.4f}\n"
        f"Max: {off_diag_abs.max():.4f}"
    )
    
    plt.annotate(
        stats_text, 
        xy=(0.02, 0.97), 
        xycoords='axes fraction',
        va='top', 
        ha='left', 
        fontsize=9, 
        bbox=dict(boxstyle='round', fc='white', alpha=0.7)
    )
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"{output_prefix}_gram_density_scale_{multiplier}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved V Gram matrix density plot to {plot_path}")

def process_single_observation(args, model, tokenizer, sliced_model, instruction, obs_idx, scalar_multipliers, offset=0):
    """Process a single observation and train DCT models with different scaling factors."""
    
    # Create output directory for this observation
    output_dir = f"{args.output_dir}_{obs_idx+offset}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse token_idxs
    token_idxs_parts = args.token_idxs.split(':')
    if len(token_idxs_parts) == 1:
        token_idxs = int(token_idxs_parts[0])
    else:
        start = None if token_idxs_parts[0] == '' else int(token_idxs_parts[0])
        end = None if token_idxs_parts[1] == '' else int(token_idxs_parts[1])
        token_idxs = slice(start, end)
    
    # Prepare chat template
    if len(args.system_prompt) > 0:
        chat_init = [{'content': args.system_prompt, 'role': 'system'}]
    else:
        chat_init = []
    
    chat = chat_init + [{'content': instruction, 'role': 'user'}]
    example = tokenizer.apply_chat_template(chat, add_special_tokens=False, tokenize=False, add_generation_prompt=True)
    
    # Tokenize just this single observation
    model_input = tokenizer([example], return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)
    seq_len = model_input["input_ids"].shape[1]

    print(f"[Observation {obs_idx}] Sequence length {seq_len}")
    
    # Create attention mask (1 for actual tokens, 0 for padding)
    attention_mask = model_input["attention_mask"].to(torch.float)
    
    d_model = model.config.hidden_size
    
    # Process the single observation
    X = torch.zeros(1, seq_len, d_model, device="cpu")
    Y = torch.zeros(1, seq_len, d_model, device="cpu")
    
    with torch.no_grad():
        input_ids = model_input["input_ids"].to(model.device)
        masks = model_input["attention_mask"].to(model.device)
        hidden_states = model(input_ids, attention_mask=masks, output_hidden_states=True).hidden_states
        h_source = hidden_states[args.source_layer_idx]  # b x t x d_model
        unsteered_target = sliced_model(h_source)  # b x t x d_model
        
        X[0, :, :] = h_source.cpu()
        Y[0, :, :] = unsteered_target.cpu()
    
    # Set up DeltaActivations with target position indices
    delta_acts_single = dct.DeltaActivations(
        sliced_model, 
        target_position_indices=token_idxs
    )
    
    # Prepare attention mask for calibration and training
    cpu_attention_mask = attention_mask.cpu()
    
    # Vectorize delta_acts for batch processing
    delta_acts = vmap(
        lambda theta, x, y, mask: delta_acts_single(theta, x, y, mask), 
        in_dims=(1, None, None, None), 
        out_dims=2,
        chunk_size=args.factor_batch_size
    )
    
    # Calibrate input scale with attention mask
    if args.scale is None:
        steering_calibrator = dct.SteeringCalibrator(target_ratio=args.target_ratio)
        
        input_scale = steering_calibrator.calibrate(
            delta_acts_single,
            X.to(delta_acts_single.device),
            Y.to(delta_acts_single.device),
            factor_batch_size=args.factor_batch_size,
            calibration_sample_size=args.calibration_sample_size,
            attention_mask=cpu_attention_mask.to(delta_acts_single.device)
        )
    else:
        input_scale = args.scale
    
    print(f"[Observation {obs_idx}] Base input scale: {input_scale}")
    
    # Train models with different scaling factors
    all_U = []
    all_V = []
    all_info = []
    
    for i, multiplier in enumerate(scalar_multipliers):
        current_scale = input_scale * multiplier
        print(f"[Observation {obs_idx}] Training with scale multiplier {multiplier}, input_scale = {current_scale}")
        
        exp_dct = dct.ExponentialDCT(num_factors=args.num_factors)
        U, V = exp_dct.fit(
            delta_acts_single, 
            X, 
            Y, 
            batch_size=args.backward_batch_size, 
            factor_batch_size=args.factor_batch_size,
            init="rand_backward",  # Only use random or rand_backward initialization
            input_scale=current_scale, 
            max_iters=args.num_iters, 
            beta=1.0,
            orthogonalize=args.ortho,
            deflation=args.deflate,
            soft_ortho_temp=args.deflation_temp,
            soft_ortho_iterations=args.deflation_iterations,
            attention_mask=cpu_attention_mask.to(delta_acts_single.device),
            separate_u=args.separate_u
        )
        
        # Save objective values
        torch.save(
            torch.tensor(exp_dct.objective_values), 
            os.path.join(output_dir, f"{args.output_prefix}_objective_values_scale_{multiplier}.pt")
        )
        save_objective_plots(args, scalar_multipliers, exp_dct, multiplier, output_dir, args.output_prefix)

        # Plot gram matrix
        plot_gram_matrix_density(V.detach().cpu(), output_dir, args.output_prefix, multiplier)
        
        all_U.append(U[:,:])
        
        # Save V normalized or scaled based on user preference
        if args.save_normalized:
            all_V.append(V[:,:])
        else:
            all_V.append(V[:,:] * current_scale)
        
        # Store info about this set of vectors
        for j in range(args.num_factors):
            all_info.append(f"scale_{multiplier}_idx_{j}")
    
    # Save all U and V matrices
    torch.save(torch.cat(all_U, dim=1), os.path.join(output_dir, f"{args.output_prefix}_U.pt"))
    torch.save(torch.cat(all_V, dim=1), os.path.join(output_dir, f"{args.output_prefix}_V.pt"))
    
    # Save vector info
    with open(os.path.join(output_dir, f"{args.output_prefix}_vector_info.json"), 'w') as f:
        json.dump(all_info, f, indent=2)
    
    # Also save a metadata file with training parameters
    metadata = vars(args)
    metadata['base_input_scale'] = input_scale
    metadata['total_vectors'] = len(all_info)
    metadata['observation_index'] = obs_idx
    metadata['observation'] = instruction
    
    with open(os.path.join(output_dir, f"{args.output_prefix}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[Observation {obs_idx}] Training complete. Saved {len(all_info)} vectors.")

def train_combined_observations(args, model, tokenizer, sliced_model, instructions, scalar_multipliers):
    """Train DCT models on combined observations (original approach)."""
    # Parse token_idxs
    token_idxs_parts = args.token_idxs.split(':')
    if len(token_idxs_parts) == 1:
        token_idxs = int(token_idxs_parts[0])
    else:
        start = None if token_idxs_parts[0] == '' else int(token_idxs_parts[0])
        end = None if token_idxs_parts[1] == '' else int(token_idxs_parts[1])
        token_idxs = slice(start, end)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare chat templates
    if len(args.system_prompt) > 0:
        chat_init = [{'content': args.system_prompt, 'role': 'system'}]
    else:
        chat_init = []
    
    chats = [chat_init + [{'content': content, 'role': 'user'}] for content in instructions]
    examples = [tokenizer.apply_chat_template(chat, add_special_tokens=False, tokenize=False, add_generation_prompt=True) for chat in chats]
    
    # Prepare dataset of unsteered activations
    model_inputs = tokenizer(examples, return_tensors="pt", padding="longest", truncation=True, max_length=args.max_length)
    max_seq_len = model_inputs["input_ids"].shape[1]
    
    # Create attention mask (1 for actual tokens, 0 for padding)
    attention_mask = model_inputs["attention_mask"].to(torch.float)
    
    d_model = model.config.hidden_size
    num_samples = min(args.num_samples, len(examples))
    
    X = torch.zeros(num_samples, max_seq_len, d_model, device="cpu")
    Y = torch.zeros(num_samples, max_seq_len, d_model, device="cpu")
    
    for t in tqdm(range(0, num_samples, args.forward_batch_size)):
        with torch.no_grad():
            input_ids = model_inputs["input_ids"][t:t+args.forward_batch_size,:].to(model.device)
            masks = model_inputs["attention_mask"][t:t+args.forward_batch_size,:].to(model.device)
            hidden_states = model(input_ids, attention_mask=masks, output_hidden_states=True).hidden_states
            h_source = hidden_states[args.source_layer_idx]  # b x t x d_model
            unsteered_target = sliced_model(h_source)  # b x t x d_model
            
            X[t:t+args.forward_batch_size, :, :] = h_source.cpu()
            Y[t:t+args.forward_batch_size, :, :] = unsteered_target.cpu()
    
    # Set up DeltaActivations with target position indices
    delta_acts_single = dct.DeltaActivations(
        sliced_model, 
        target_position_indices=token_idxs
    )
    
    # Prepare attention mask for calibration and training
    cpu_attention_mask = attention_mask.cpu()
    
    # Vectorize delta_acts for batch processing
    delta_acts = vmap(
        lambda theta, x, y, mask: delta_acts_single(theta, x, y, mask), 
        in_dims=(1, None, None, None), 
        out_dims=2,
        chunk_size=args.factor_batch_size
    )
    
    # Calibrate input scale with attention mask
    if args.scale is None:
        steering_calibrator = dct.SteeringCalibrator(target_ratio=args.target_ratio)
        
        input_scale = steering_calibrator.calibrate(
            delta_acts_single,
            X.to(delta_acts_single.device),
            Y.to(delta_acts_single.device),
            factor_batch_size=args.factor_batch_size,
            calibration_sample_size=args.calibration_sample_size,
            attention_mask=cpu_attention_mask.to(delta_acts_single.device)
        )
    else:
        input_scale = args.scale
    
    print(f"Base input scale: {input_scale}")
    
    # Train models with different scaling factors
    all_U = []
    all_V = []
    all_info = []
    
    for i, multiplier in enumerate(scalar_multipliers):
        current_scale = input_scale * multiplier
        print(f"Training with scale multiplier {multiplier}, input_scale = {current_scale}")
        
        exp_dct = dct.ExponentialDCT(num_factors=args.num_factors)
        U, V = exp_dct.fit(
            delta_acts_single, 
            X, 
            Y, 
            batch_size=args.backward_batch_size, 
            factor_batch_size=args.factor_batch_size,
            init="rand_backward",  # Only use random or rand_backward initialization
            input_scale=current_scale, 
            max_iters=args.num_iters, 
            beta=1.0,
            orthogonalize=args.ortho,
            deflation=args.deflate,
            soft_ortho_temp=args.deflation_temp,
            soft_ortho_iterations=args.deflation_iterations,
            attention_mask=cpu_attention_mask.to(delta_acts_single.device),
            separate_u=args.separate_u
        )
        
        # Save objective values
        torch.save(
            torch.tensor(exp_dct.objective_values), 
            os.path.join(args.output_dir, f"{args.output_prefix}_objective_values_scale_{multiplier}.pt")
        )
        save_objective_plots(args, scalar_multipliers, exp_dct, multiplier, args.output_dir, args.output_prefix)

        # Plot gram matrix
        plot_gram_matrix_density(V.detach().cpu(), args.output_dir, args.output_prefix, multiplier)
        
        all_U.append(U[:,:])
        
        # Save V normalized or scaled based on user preference
        if args.save_normalized:
            all_V.append(V[:,:])
        else:
            all_V.append(V[:,:] * current_scale)
        
        # Store info about this set of vectors
        for j in range(args.num_factors):
            all_info.append(f"scale_{multiplier}_idx_{j}")
    
    # Save all U and V matrices
    torch.save(torch.cat(all_U, dim=1), os.path.join(args.output_dir, f"{args.output_prefix}_U.pt"))
    torch.save(torch.cat(all_V, dim=1), os.path.join(args.output_dir, f"{args.output_prefix}_V.pt"))
    
    # Save vector info
    with open(os.path.join(args.output_dir, f"{args.output_prefix}_vector_info.json"), 'w') as f:
        json.dump(all_info, f, indent=2)
    
    # Also save a metadata file with training parameters
    metadata = vars(args)
    metadata['base_input_scale'] = input_scale
    metadata['total_vectors'] = len(all_info)
    
    with open(os.path.join(args.output_dir, f"{args.output_prefix}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training complete. Saved {len(all_info)} vectors.")

def main():
    args = parse_args()
    
    # Parse scalar multipliers
    scalar_multipliers = [float(x) for x in args.scalar_multipliers.split(',')]
    
    # Set device
    torch.set_default_device(args.device)
    torch.manual_seed(325)
    
    # Load dataset and get instructions
    dataset = load_from_disk(args.dataset + f"/{args.split}" if args.split else args.dataset)
    # Use the proper Dataset select method
    dataset_subset = dataset.select(range(min(args.num_samples, len(dataset))))
    if args.data_start is not None:
        dataset_subset = dataset_subset.select(range(args.data_start, args.data_end))
    instructions = dataset_subset[args.field]
    
    print(f"Loaded {len(instructions)} instructions from dataset")
    
    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Use tokenizer_name if provided, otherwise fall back to model_name
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, 
        trust_remote_code=True, 
        padding_side="left",
        truncation_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device,
        trust_remote_code=True
    )
    for p in model.parameters():
        p.requires_grad = False
    
    # Slice model
    # Verify that slice works correctly
    model_inputs = tokenizer(["colourless green sheep sleep furiously"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        hidden_states = model(model_inputs["input_ids"], output_hidden_states=True).hidden_states
    
    test_slice = dct.SlicedModel(model, start_layer=3, end_layer=5, layers_name="model.layers")
    with torch.no_grad():
        if "gemma2" in str(type(model)):
            rtol = 1e-2
        else:
            rtol = 1e-5
        if type(model).__name__ == 'Qwen3MoeForCausalLM':
            assert torch.allclose(test_slice(hidden_states[3]), hidden_states[6], rtol=rtol)
        else:
            assert torch.allclose(test_slice(hidden_states[3]), hidden_states[5], rtol=rtol)
    
    # Create the actual slice we'll use
    sliced_model = dct.SlicedModel(
        model, 
        start_layer=args.source_layer_idx, 
        end_layer=args.target_layer_idx, 
        layers_name="model.layers"
    )
    
    if args.for_each:
        # Process each observation separately
        for idx, instruction in enumerate(instructions):
            print(f"\n=== Processing observation {idx} ===")
            process_single_observation(args, model, tokenizer, sliced_model, instruction, idx, scalar_multipliers, offset=args.data_start or 0)
    else:
        # Process all observations together (original approach)
        train_combined_observations(args, model, tokenizer, sliced_model, instructions, scalar_multipliers)

if __name__=="__main__":
    main()