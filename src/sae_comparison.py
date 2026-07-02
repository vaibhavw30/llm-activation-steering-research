#!/usr/bin/env python
# coding: utf-8

import torch
from torch.nn import functional as F
import seaborn as sns
from matplotlib import pyplot as plt
from sae_lens import SAE  # pip install sae-lens
import matplotlib

matplotlib.rcParams.update({
    "font.family": "serif",  # or "DejaVu Serif", etc.
    "pdf.fonttype": 42,      # Embed TrueType fonts (Type 42)
    "ps.fonttype": 42,       # Same for EPS
})

# Set default device
torch.set_default_device("cuda")

# Define SAE configurations to compare
sae_configs = [
    {
        "release": "gemma-scope-9b-pt-res-canonical",
        "sae_id": "layer_9/width_16k/canonical",
        "name": "16K"
    },
    {
        "release": "gemma-scope-9b-pt-res-canonical",  # replace with actual second SAE
        "sae_id": "layer_9/width_131k/canonical",       # replace with actual second SAE
        "name": "131K"
    },
    {
        "release": "gemma-scope-9b-pt-res-canonical",  # replace with actual third SAE
        "sae_id": "layer_9/width_1m/canonical",       # replace with actual third SAE
        "name": "1m"
    }
]

# Load the DCT basis once (assuming it's the same for all comparisons)
with torch.no_grad():
    V = torch.load("../runs/train/story/gemma2_9b/_V.pt")
    V = F.normalize(V, dim=0)

# Create figure for the plot
plt.figure(figsize=(12, 8))

# Process each SAE
for config in sae_configs:
    # Load SAE model
    print(f"Loading {config['name']}...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=config["release"],
        sae_id=config["sae_id"],
    )
    
    # Get normalized decoder weights
    with torch.no_grad():
        W = sae.W_dec.clone().detach().t().cuda()
        W = F.normalize(W, dim=0)
    
    # Compute basis-aligned dots
    I = torch.eye(V.shape[0], V.shape[1])
    basis_aligned_dots = (I.t() @ W).max(dim=1).values.cpu()
    
    # Compute DCT dots
    dct_dots = (V.t() @ W).max(dim=1).values.cpu()
    
    # Compute random dots
    R = F.normalize(torch.randn(V.shape[0], V.shape[1]), dim=0)
    random_dots = (R.t() @ W).max(dim=1).values.cpu()
    
    # Plot distributions
    sns.kdeplot(dct_dots, label=f'DCT - {config["name"]}', linestyle='-')
    sns.kdeplot(random_dots, label=f'Random - {config["name"]}', linestyle='--')
    
    # Optional: Print sorted DCT dots
    print(f"Top DCT dots for {config['name']}:")
    sorted_dct = dct_dots.sort(descending=True)[0]
    print(sorted_dct[:10])  # Print top 10 values

# Add plot styling
plt.title('Alignment between SAE and DCT Feature Directions')
plt.xlabel('Maximum Dot Product')
plt.ylabel('Density')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig('sae_comparison_results.pdf', bbox_inches="tight")
plt.show()