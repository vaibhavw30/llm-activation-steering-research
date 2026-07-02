import functools
from torch import nn, vmap
import torch
from torch.func import vjp, jvp, grad
from torch.nn import functional as F
from tqdm import tqdm
import math
from scipy.optimize import root_scalar
from typing import Any, Callable, Optional, Union, Tuple
from functools import wraps


def rgetattr(obj, path):
    return functools.reduce(getattr, path.split("."), obj)

def rhasattr(obj, path):
    try:
        functools.reduce(hasattr, path.split("."), obj)
        return True
    except (AttributeError, TypeError):
        return False

def vmap_with_progress(
    func: Callable,
    in_dims: Union[int, None, Any] = 0,
    out_dims: Union[int, Tuple[int, ...]] = 0,
    randomness: str = 'error',
    chunk_size: Optional[int] = None,
    progress_kwargs: Optional[dict] = None
) -> Callable:
    """
    A wrapper around torch.func.vmap that displays a progress bar when using chunking.
    Supports all vmap parameters including in_dims and out_dims.
    
    Args:
        func: A Python function that takes one or more arguments.
        in_dims: Specifies which dimension of the inputs should be mapped over.
        out_dims: Specifies where the mapped dimension should appear in the outputs.
        randomness: Specifies randomness behavior ('error', 'same', or 'different').
        chunk_size: If not None, compute the vmap chunk_size samples at a time.
        progress_kwargs: Dict of kwargs to pass to tqdm progress bar.
    
    Returns:
        A vectorized function that shows progress when chunk_size is set.
    """
    # Create the original vmap function
    original_vmap = vmap(func, in_dims=in_dims, out_dims=out_dims, randomness=randomness)
    
    # If no chunk_size is specified, use the built-in chunk_size parameter
    if chunk_size is None:
        return original_vmap
    
    # Otherwise, handle chunking with progress bar manually
    if progress_kwargs is None:
        progress_kwargs = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Parse in_dims to figure out which args have batch dimensions
        if isinstance(in_dims, int):
            batched_dims = [in_dims] * len(args)
        elif in_dims is None:
            batched_dims = [None] * len(args)
        elif isinstance(in_dims, (tuple, list)):
            batched_dims = list(in_dims)
            # Pad with defaults if needed
            batched_dims.extend([0] * (len(args) - len(batched_dims)))
        else:
            # Handle nested structures - for simplicity, we'll require explicit handling
            raise NotImplementedError("Complex nested in_dims structures not yet supported in progress version")
        
        # Find the batch size from the first batched argument
        batch_size = None
        for arg, dim in zip(args, batched_dims):
            if dim is not None and hasattr(arg, 'shape'):
                batch_size = arg.shape[dim]
                break
        
        if batch_size is None:
            # No batched arguments found, just call the original function
            return original_vmap(*args, **kwargs)
        
        num_chunks = math.ceil(batch_size / chunk_size)
        results = []
        
        # Process in chunks with progress bar
        pbar = tqdm(
            range(0, batch_size, chunk_size),
            total=num_chunks,
            desc=progress_kwargs.get('desc', f'Processing {num_chunks} chunks'),
            **{k: v for k, v in progress_kwargs.items() if k != 'desc'}
        )
        
        for i in pbar:
            end_idx = min(i + chunk_size, batch_size)
            
            # Extract chunk for each argument
            chunk_args = []
            for arg, dim in zip(args, batched_dims):
                if dim is None:
                    # This argument is not batched
                    chunk_args.append(arg)
                else:
                    # Extract slice along the specified dimension
                    if hasattr(arg, 'shape'):
                        # It's a tensor
                        slices = [slice(None)] * len(arg.shape)
                        slices[dim] = slice(i, end_idx)
                        chunk_args.append(arg[tuple(slices)])
                    else:
                        # Handle other types (lists, tuples, etc.)
                        chunk_args.append(arg[i:end_idx])
            
            # Process chunk
            chunk_result = original_vmap(*chunk_args, **kwargs)
            results.append(chunk_result)
        
        # Concatenate results along the output dimension(s)
        if isinstance(results[0], torch.Tensor):
            # Single output tensor
            cat_dim = out_dims if isinstance(out_dims, int) else 0
            return torch.cat(results, dim=cat_dim)
        elif isinstance(results[0], (tuple, list)):
            # Multiple outputs
            concatenated = []
            output_dims = out_dims if isinstance(out_dims, (tuple, list)) else [out_dims] * len(results[0])
            for j in range(len(results[0])):
                cat_dim = output_dims[j] if j < len(output_dims) else 0
                concatenated.append(torch.cat([r[j] for r in results], dim=cat_dim))
            return type(results[0])(concatenated)
        else:
            # Fallback for other types
            return torch.cat(results, dim=0)
    
    return wrapper

def soft_ortho(V, num_iterations=1, temperature=1.0, logit_bias=None):
    """
    Performs soft orthogonalization of vectors in V using vectorized operations.
    
    Args:
        V: Tensor of shape (d, n) where d is the dimension and n is the number of vectors
        num_iterations: Number of times to repeat the process
        temperature: Temperature parameter for the softmax (higher = softer)
        logit_bias: Optional tensor of shape (n) to add as bias in the softmax
        
    Returns:
        Updated tensor of the same shape
    """
    d, n = V.shape
    
    # Initialize logit_bias if not provided
    if logit_bias is None:
        logit_bias = torch.zeros(n, device=V.device)
    
    # Normalize columns initially
    V = F.normalize(V, p=2, dim=0)
    
    # Create masks for excluding self-interactions
    mask = torch.ones(n, n, device=V.device) - torch.eye(n, device=V.device)
    
    for _ in range(num_iterations):
        # Compute all pairwise dot products
        dot_products = torch.matmul(V.T, V) / temperature  # Shape: (n, n)
        
        # Add logit bias (broadcasting to each row)
        dot_products = dot_products + logit_bias.unsqueeze(0)
        
        # Prepare for softmax by setting diagonal (self-interactions) to -inf
        dot_products = dot_products.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to each row to get weights
        weights = F.softmax(dot_products, dim=1)  # Shape: (n, n)
        
        # Compute the weighted sums for all vectors at once
        weighted_sums = torch.matmul(V, weights.T)  # Shape: (d, n)
        
        # Update all vectors at once
        V_new = V - weighted_sums
        
        # Normalize the columns
        V_new = F.normalize(V_new, p=2, dim=0)
        
        V = V_new
    
    return V



class SlicedModel(nn.Module):
    def __init__(self, model, start_layer, end_layer, layers_name=None):
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer
        if layers_name is None:
            if hasattr(self.model, "layers"):  
                self.layers_name = "model.layers"
            elif hasattr(self.model, "model"): 
                self.layers_name =  "model.model.layers"
            else:
                raise ValueError(f"don't know how to get layer list for {type(model)}")
        else:
            self.layers_name = layers_name
        self.layers = rgetattr(self.model, self.layers_name)
        self.layers_name_split = self.layers_name.split(".")
    def reset(self):
        setattr(self.model.config, "num_hidden_layers",self.depth)
        setattr(rgetattr(self.model, ".".join(self.layers_name_split[:-1])), self.layers_name_split[-1], self.L)
        for i in range(len(rgetattr(self.model, self.layers_name))):
            rgetattr(self.model, self.layers_name)[i].self_attn.layer_idx = i
        pass


    def forward(self, h, past_key_values=None):
        if past_key_values is not None:
            cache_depth = len(past_key_values.key_cache)
        # mutate model so that forward pass only runs the specified middle layers
        self.L = self.model.model.layers
        self.depth = self.model.config.num_hidden_layers
        self.model.model.layers = self.L[self.start_layer:self.end_layer+2]
        setattr(self.model.config, "num_hidden_layers", self.end_layer-self.start_layer+1)
        for i, layer in enumerate(self.model.model.layers):
            layer.self_attn.layer_idx = i
        if past_key_values:
            cache_len = past_key_values.key_cache[0].shape[2]

        # if Gemma2 model then scale the hidden states
        if "gemma2" in str(type(self.model)):
            h_dtype = h.dtype
            h = h.to(torch.float32) / self.model.config.hidden_size**0.5
            h = h.to(h_dtype)

        # actually run the forward pass
        next_output = self.model(
            inputs_embeds=h,
            past_key_values=past_key_values,
            use_cache=False,
            output_hidden_states=True
        )

        # reset cache
        if past_key_values is not None:
            with torch.no_grad():
                past_key_values.key_cache = past_key_values.key_cache[:cache_depth]
                past_key_values.value_cache = past_key_values.value_cache[:cache_depth]
                for i, key in enumerate(past_key_values.key_cache):
                    past_key_values.key_cache[i] = key[:,:,:cache_len,:]
                for i, value in enumerate(past_key_values.value_cache):
                    past_key_values.value_cache[i] = value[:,:,:cache_len,:]
        # reset model to un-mutated state
        self.reset()
        
        return next_output.hidden_states[-2]

class DeltaActivations(nn.Module):
    def __init__(self, sliced_model, target_position_indices=slice(-3,None)):
        super().__init__()
        self.sliced_model = sliced_model
        self.device = sliced_model.model.device
        self.target_position_indices = target_position_indices
    
    def forward(self, theta, x, y, attention_mask=None):
        '''
        computes average delta in target layer activations as a 
        function of bias theta
        
        Args:
            theta: bias to add to input activations
            x: input activations
            y: target activations to compare against
            attention_mask: optional attention mask (batch_size x seq_len) with 1 for valid tokens, 0 for padding
        
        Returns:
            Average delta across specified target positions
        '''
        delta = self.sliced_model(x+theta) - y # batch_size x seq_len x d_model
        delta = delta[:, self.target_position_indices, :]
        
        # Apply attention mask if available
        if attention_mask is not None:
            # Extract the batch indices for the current batch
            batch_indices = list(range(delta.shape[0]))
            
            # Select the same target positions from the attention mask
            if isinstance(self.target_position_indices, slice):
                start = self.target_position_indices.start or 0
                stop = self.target_position_indices.stop or attention_mask.shape[1]
                if stop < 0:
                    stop = attention_mask.shape[1] + stop
                mask_slice = attention_mask[batch_indices, start:stop]
            else:
                mask_slice = attention_mask[batch_indices, self.target_position_indices]
            
            # Expand mask to match delta dimensions
            mask = mask_slice.unsqueeze(-1).to(self.device)
            
            # Apply mask to delta (zero out padding tokens)
            delta = delta * mask
            
            # Sum and divide by the number of non-padding tokens (per batch item)
            token_counts = mask_slice.sum(dim=1, keepdim=True).to(self.device)
            # Avoid division by zero
            token_counts = torch.clamp(token_counts, min=1.0)
            
            # Take the mean across valid tokens only
            return (delta.sum(dim=1) / token_counts)
        
        # No mask, just take the mean across all tokens in the target positions
        return delta.mean(dim=1)

class StreamingAverage:
    """
    Maintains a streaming average of tensors.
    Handles variable batch sizes and arbitrary tensor dimensions.
    """
    def __init__(self):
        self.count = 0
        self.mean = None
    
    def update(self, batch: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Updates the streaming average with a new batch of data.
        
        Args:
            batch: Tensor of shape (batch_size, dim1, ..., dimk)
                  The first dimension is assumed to be the batch dimension
            mask: Optional tensor of same shape as batch or broadcastable to it
                  1 for valid values, 0 for padding/masked values
        
        Returns:
            Current mean after incorporating the new batch
        """
        if mask is not None:
            # Compute masked batch mean
            masked_batch = batch * mask
            batch_sum = masked_batch.sum(dim=0)
            mask_sum = mask.sum(dim=0)
            batch_mean = batch_sum / mask_sum.clamp(min=1.0)  # Avoid division by zero
            batch_size = mask.sum().item()
        else:
            batch_size = batch.size(0)
            batch_mean = batch.mean(dim=0)
        
        if self.mean is None:
            # First batch - initialize mean
            self.mean = batch_mean
            self.count = batch_size
            return self.mean
        
        # Update count
        new_count = self.count + batch_size
        
        # Update mean using formula:
        # new_mean = old_mean + (batch_mean - old_mean) * (batch_size / new_count)
        self.mean = self.mean + (batch_mean - self.mean) * (batch_size / new_count)
        self.count = new_count
        
        return self.mean
    
    def get_mean(self) -> torch.Tensor:
        """Returns the current mean."""
        if self.mean is None:
            raise ValueError("No data has been processed yet")
        return self.mean
    
    def reset(self):
        """Resets the streaming average."""
        self.count = 0
        self.mean = None

class SteeringCalibrator():
    def __init__(self, target_ratio=.5):
        self.target_ratio=target_ratio
    def calibrate(self, delta_acts_single, X, Y, batch_size=1,
                  calibration_sample_size=30, factor_batch_size=16, attention_mask=None):
        """
        Calibrate the input scale for steering vectors.
        
        Args:
            delta_acts_single: Function computing change in target activations
            X: Source activations (batch_size, seq_len, d_model)
            Y: Target activations (batch_size, seq_len, d_model)
            batch_size: Batch size for processing samples
            calibration_sample_size: Number of random directions to use
            factor_batch_size: Batch size for processing factors
            attention_mask: Optional attention mask (batch_size, seq_len)
            
        Returns:
            Calibrated input scale value
        """
        delta_acts = vmap(lambda theta, x, y, mask: delta_acts_single(theta, x, y, mask), 
                          in_dims=(1, None, None, None), 
                          out_dims=2,
                          chunk_size=factor_batch_size)
        
        d_model = X.shape[2]
        V_cal = F.normalize(torch.randn(d_model, calibration_sample_size, device=delta_acts_single.device), dim=0)
        
        def jvp_single(v, X, Y, mask):
            v0 = torch.zeros_like(v)
            _, jvp_out = jvp(lambda _v: delta_acts_single(_v, X, Y, mask), (v0,), (v,))
            return jvp_out
        
        jvp_batch = vmap(lambda v, X, Y, mask: jvp_single(v, X, Y, mask), 
                         in_dims=(1, None, None, None), 
                         out_dims=(2), 
                         chunk_size=factor_batch_size)
    
        U_cal_avg = StreamingAverage()
        with torch.no_grad():
            for b in range(0, X.shape[0], batch_size):
                x = X[b:b+batch_size,:,:].to(delta_acts_single.device)
                y = Y[b:b+batch_size,:,:].to(delta_acts_single.device)
                mask = None if attention_mask is None else attention_mask[b:b+batch_size,:].to(delta_acts_single.device)
                
                # Pass the mask to JVP
                U_cal_batch = jvp_batch(V_cal, x, y, mask)
                U_cal_avg.update(U_cal_batch)
                
        U_cal = U_cal_avg.get_mean()
        U_cal_norms = U_cal.norm(dim=0)
    
        def jacobian_ratio(r):
            denom = (r*U_cal_norms).pow(2)
            delta_acts_avg = StreamingAverage()
            with torch.no_grad():
                for b in range(0, X.shape[0], batch_size):
                    x = X[b:b+batch_size,:,:].to(delta_acts_single.device)
                    y = Y[b:b+batch_size,:,:].to(delta_acts_single.device)
                    mask = None if attention_mask is None else attention_mask[b:b+batch_size,:].to(delta_acts_single.device)
                    
                    # Pass the mask to delta_acts
                    delta_acts_batch = delta_acts(r*V_cal, x, y, mask)
                    delta_acts_avg.update(delta_acts_batch)
                    
            num = (delta_acts_avg.get_mean()-r*U_cal).pow(2).sum(dim=0)
            return math.sqrt((num / denom).mean())
    
        # solve for jacobian_ratio = target_ratio
        soln = root_scalar(lambda r: jacobian_ratio(r)-self.target_ratio, bracket=[.01, 100.0])
        self.R = soln.root
        return self.R

class ExponentialDCT():
    def __init__(self, num_factors=512):
        self.num_factors = num_factors
        
    def _init_rand(self, delta_acts, X, Y, attention_mask=None):
        print("initializing V,U randomly...")
        # initialize V randomly
        self.V = F.normalize(torch.randn(self.d_source, self.num_factors, device=self.device), dim=0)
        self.U = F.normalize(torch.randn(self.d_target, self.num_factors, device=self.device), dim=0)
        pass
    
    def _init_rand_backward(self, delta_acts_single, X, Y, attention_mask=None):
        print("initializing U randomly and V with gradients...")
        # Initialize U randomly
        self.U = F.normalize(torch.randn(self.d_target, self.num_factors, device=self.device), dim=0)
        
        # Compute V as gradients of u_i'*delta_acts at v0 = 0
        v0 = torch.zeros(self.d_source, device=self.device)
        
        # Define a function to compute the gradient for a single u direction
        def grad_for_u(u, x, y, mask):
            def u_dot_delta(v):
                delta = delta_acts_single(v, x, y, mask)
                return (delta @ u).mean()
            return grad(u_dot_delta)(v0).detach()
        
        # Initialize V with the right shape
        self.V = torch.zeros(self.d_source, self.num_factors, device=self.device)
        
        # For large num_factors, we'll compute the Jacobian once and then project
        if self.num_factors > self.d_source:
            print("Computing Jacobian for efficient backward initialization...")
            # Compute the Jacobian at v0 = 0
            J_avg = StreamingAverage()
            
            # Function to compute Jacobian-vector product
            def jvp_single(v, x, y, mask):
                v0_zeros = torch.zeros_like(v)
                _, jvp_out = jvp(lambda _v: delta_acts_single(_v, x, y, mask), (v0_zeros,), (v,))
                return jvp_out
            
            # Batch over standard basis vectors
            jvp_batch = vmap_with_progress(lambda v, x, y, mask: jvp_single(v, x, y, mask), 
                              in_dims=(1, None, None, None), 
                              out_dims=2, 
                              chunk_size=self.factor_batch_size)
            
            # Use standard basis to compute full Jacobian
            V_in = torch.eye(self.d_source, device=self.device)
            
            with torch.no_grad():
                for t in tqdm(range(0, X.shape[0], self.batch_size)):
                    batch_slice = slice(t, t + min(self.batch_size, X.shape[0] - t))
                    x_full = X[batch_slice].to(self.device)
                    y_full = Y[batch_slice].to(self.device)
                    
                    # Handle attention mask for this batch
                    if attention_mask is not None:
                        batch_mask = attention_mask[batch_slice].to(self.device)
                        
                        # Find the right-most non-masked token for each sample in the batch
                        right_most_indices = torch.argmax(
                            # Multiply position indices by mask to get 0 for masked positions
                            torch.arange(batch_mask.shape[1], device=self.device).unsqueeze(0) * batch_mask,
                            dim=1
                        )
                        
                        # Find the left-most of all right-most indices to determine where to start
                        # This ensures we include all tokens needed by every sample
                        left_bound = right_most_indices.min().item()
                        
                        # Subset the data to only include tokens from left_bound onwards
                        x = x_full[:, left_bound:, :]
                        y = y_full[:, left_bound:, :]
                        
                        # Update the mask accordingly
                        mask = batch_mask[:, left_bound:]
                    else:
                        # If no mask, use the full sequence
                        x = x_full
                        y = y_full
                        mask = None
                    
                    J_batch = jvp_batch(V_in, x, y, mask)
                    J_avg.update(J_batch)
            
            J = J_avg.get_mean()  # shape: [d_target, d_source]
            
            # Compute V as J.T @ U
            for i in range(self.num_factors):
                self.V[:, i] = J.t() @ self.U[:, i]
        else:
            # For smaller num_factors, directly compute gradients for each u_i
            print("Computing gradients for each direction...")
            
            # Batch compute gradients
            grad_batch = vmap(lambda u, x, y, mask: grad_for_u(u, x, y, mask), 
                              in_dims=(1, None, None, None), 
                              out_dims=1, 
                              chunk_size=self.factor_batch_size)
            
            with torch.no_grad():
                V_avg = StreamingAverage()
                for b in tqdm(range(0, X.shape[0], self.batch_size)):
                    batch_slice = slice(b, b + min(self.batch_size, X.shape[0] - b))
                    x_full = X[batch_slice].to(self.device)
                    y_full = Y[batch_slice].to(self.device)
                    
                    # Handle attention mask for this batch
                    if attention_mask is not None:
                        batch_mask = attention_mask[batch_slice].to(self.device)
                        
                        # Find the right-most non-masked token for each sample in the batch
                        right_most_indices = torch.argmax(
                            # Multiply position indices by mask to get 0 for masked positions
                            torch.arange(batch_mask.shape[1], device=self.device).unsqueeze(0) * batch_mask,
                            dim=1
                        )
                        
                        # Find the left-most of all right-most indices to determine where to start
                        # This ensures we include all tokens needed by every sample
                        left_bound = right_most_indices.min().item()
                        
                        # Subset the data to only include tokens from left_bound onwards
                        x = x_full[:, left_bound:, :]
                        y = y_full[:, left_bound:, :]
                        
                        # Update the mask accordingly
                        mask = batch_mask[:, left_bound:]
                    else:
                        # If no mask, use the full sequence
                        x = x_full
                        y = y_full
                        mask = None
                    
                    V_batch = grad_batch(self.U, x, y, mask)
                    V_avg.update(V_batch.unsqueeze(0))
                self.V = V_avg.get_mean()
        
        # Normalize V
        self.V = F.normalize(self.V, dim=0)
    
    def rank(self, delta_acts_single, X, Y, target_vec=None, batch_size=1, factor_batch_size=16, attention_mask=None):
        """
        Rank the factors based on their contribution to the delta activations.
        
        Args:
            delta_acts_single: Function computing change in target activations
            X: Source activations (n_samples, seq_len, d_source)
            Y: Target activations (n_samples, seq_len, d_target)
            target_vec: Optional target vector to project deltas onto
            batch_size: Batch size for processing samples
            factor_batch_size: Batch size for processing factors
            attention_mask: Optional attention mask (n_samples, seq_len)
            
        Returns:
            Tuple of (scores, indices) where scores are sorted in descending order
        """
        delta_acts = vmap(
            lambda theta, x, y, mask: delta_acts_single(theta, x, y, mask), 
            in_dims=(1, None, None, None), 
            out_dims=2,
            chunk_size=factor_batch_size
        )
        
        num_samples = X.shape[0]
        Delta_avg = StreamingAverage()
        
        with torch.no_grad():
            for b in tqdm(range(0, num_samples, batch_size)):
                x = X[b:b+batch_size,:,:].to(self.device)
                y = Y[b:b+batch_size,:,:].to(self.device)
                mask = None if attention_mask is None else attention_mask[b:b+batch_size,:].to(self.device)
                
                Delta_batch = delta_acts(self.input_scale * self.V, x, y, mask)              
                Delta_avg.update(Delta_batch)
                
            if target_vec is None:
                self.alphas = (Delta_avg.get_mean() * self.U).sum(dim=0)
                K = (self.U.t() @ self.U) * torch.expm1(self.V.t() @ self.V)
                self.alphas = torch.linalg.solve(K, self.alphas)
                self.scores = self.alphas.pow(2)
                self.scores, self.indices = torch.sort(self.scores, descending=True)
            else:
                self.scores = Delta_avg.get_mean().t() @ target_vec.to(self.device)
                self.scores, self.indices = torch.sort(self.scores, descending=True)                
        
        return self.scores, self.indices        
    def fit(self, delta_acts_single, X, Y, batch_size=1, factor_batch_size=16, init="random", 
            input_scale=1.0, max_iters=10, beta=1.0, orthogonalize=True, deflation=False, 
            soft_ortho_temp=1.0, soft_ortho_iterations=10, attention_mask=None, separate_u=False):
        '''Fit DCT
        
        Parameters
        ----------
        delta_acts_single : function computing change in target activations as a function of source-layer bias
        (theta, x, y, attention_mask)
    
        X (tensor) : tensor of source activations (n_samples, seq_len, d_source)
        
        Y (tensor) : tensor of target activations (n_samples, seq_len, d_target)
        
        batch_size (int) : batch size over samples
        
        factor_batch_size (int) : batch size over factors
    
        init (string): initialization strategy {"random", "rand_backward"}
    
        input_scale (float) : norm of steering vector in source layer
    
        max_iters (int) : max iters
    
        beta (float) : default = 1.0, set smaller for more stable training
        
        orthogonalize (bool) : whether to apply hard orthogonalization using QR decomposition
        
        deflation (bool) : whether to apply soft orthogonalization with bias based on G_V column norms
        
        soft_ortho_temp (float) : temperature parameter for soft orthogonalization
        
        soft_ortho_iterations (int) : number of iterations for soft orthogonalization
        
        attention_mask (tensor) : optional attention mask (n_samples, seq_len) with 1 for valid tokens, 0 for padding
        
        separate_u (bool) : whether to let output directions vary across data-points
        '''
        assert(init in ["random", "rand_backward"])
        self.num_samples, self.seq_len, self.d_source = X.shape
        _, _, self.d_target = Y.shape
        self.batch_size = batch_size
        self.factor_batch_size = factor_batch_size
        self.device = delta_acts_single.device
        self.input_scale = input_scale
        self.max_iters = max_iters
        self.beta = beta
        
        # Create vectorized delta_acts function that passes attention_mask
        delta_acts = vmap(
            lambda theta, x, y, mask: delta_acts_single(theta, x, y, mask), 
            in_dims=(1, None, None, None), 
            out_dims=2,
            chunk_size=factor_batch_size
        )
    
        # init
        if init == "random":
            self._init_rand(delta_acts, X, Y, attention_mask)
        elif init == "rand_backward":
            self._init_rand_backward(delta_acts_single, X, Y, attention_mask)
    
        # vjp helper functions that include attention_mask
        def vjp_single(u, v, X, Y, mask):
            output, vjp_fn = vjp(lambda _v: delta_acts_single(_v, X, Y, mask), v)
            with torch.no_grad():
                udots = output @ u
            return udots, output.detach(), vjp_fn(u.expand(X.shape[0], -1))[0].detach()
            
        vjp_batch = vmap(
            lambda u, v, X, Y, mask: vjp_single(u, v, X, Y, mask), 
            in_dims=(1, 1, None, None, None),
            out_dims=(1, 2, 1), 
            chunk_size=self.factor_batch_size
        )
    
        # main training loop
        self.U = nn.Parameter(self.U)
        self.V = nn.Parameter(self.V)
        fdots = []
        penalties = []
        objective_values = []
        # Track G_V norms for each factor
        gv_norms_history = []
        
        print("training...")
        for i in tqdm(range(self.max_iters)):
            # orthogonalize
            if orthogonalize and not deflation:
                with torch.no_grad():
                    self.V.data, _ = torch.linalg.qr(self.V)
    
            # loop over data to compute updates
            fdot_avg = StreamingAverage()
            G_U_avg = StreamingAverage()
            G_V_avg = StreamingAverage()
            
            for b in tqdm(range(0, self.num_samples, self.batch_size)):
                batch_slice = slice(b, b + min(self.batch_size, self.num_samples - b))
                x_full = X[batch_slice].to(self.device)
                y_full = Y[batch_slice].to(self.device)
                
                # Get attention mask for this batch
                if attention_mask is not None:
                    batch_mask = attention_mask[batch_slice].to(self.device)
                    
                    # Find the right-most non-masked token for each sample in the batch
                    right_most_indices = torch.argmax(
                        # Multiply position indices by mask to get 0 for masked positions
                        torch.arange(batch_mask.shape[1], device=self.device).unsqueeze(0) * batch_mask,
                        dim=1
                    )
                    
                    # Find the left-most of all right-most indices to determine where to start
                    # This ensures we include all tokens needed by every sample
                    left_bound = right_most_indices.min().item()
                    
                    # Subset the data to only include tokens from left_bound onwards
                    x = x_full[:, left_bound:, :]
                    y = y_full[:, left_bound:, :]
                    
                    # Update the attention mask for delta_acts_single accordingly
                    delta_acts_single.attention_mask = batch_mask[:, left_bound:]
                else:
                    # If no mask, use the full sequence
                    x = x_full
                    y = y_full
                    batch_mask=None
    
                if separate_u:
                    self.V.grad = None
                    gub = delta_acts(self.input_scale*self.V, x, y, batch_mask) # batch_size x d_target x num_factors
                    fb = gub.norm(dim=1, p=2) # batch_size x num_factors
                    fb.sum().backward()
                    with torch.no_grad():
                        gvb = self.V.grad.detach() 
                else:
                    with torch.no_grad():
                        fb, gub, gvb = vjp_batch(self.U, self.input_scale*self.V, x, y, batch_mask)
                with torch.no_grad():
                    G_U_avg.update(gub)
                    G_V_avg.update(gvb.unsqueeze(0))
                    fdot_avg.update(fb.mean(1).unsqueeze(1))
                    
            fdot_all = fdot_avg.get_mean()[0].item()
            fdots.append(fdot_all)
            G_U = G_U_avg.get_mean()
            G_V = G_V_avg.get_mean()
            
            # Calculate the norms of the G_V columns
            with torch.no_grad():
                G_V_norms = torch.norm(G_V, dim=0)  # Shape: [num_factors]
                gv_norms_history.append(G_V_norms.clone())
        
            # update
            with torch.no_grad():
                self.U.data = F.normalize(self.beta*G_U+(1-self.beta)*self.U.data, dim=0)
                self.V.data = F.normalize(self.beta*G_V+(1-self.beta)*self.V.data, dim=0)
                
                # Apply soft orthogonalization with deflation if enabled
                if deflation:
                    # Use log of G_V column norms as logit bias
                    logit_bias = torch.log(G_V_norms + 1e-8)
                    self.V.data = soft_ortho(
                        self.V.data, 
                        num_iterations=soft_ortho_iterations,
                        temperature=soft_ortho_temp,
                        logit_bias=logit_bias
                    )
                    
                objective_values.append(fdot_all)
    
        self.objective_values = objective_values
        self.gv_norms_history = gv_norms_history
        
        return self.U, self.V