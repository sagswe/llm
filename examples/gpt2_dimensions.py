"""
How Linear layer dimensions are deduced from model architecture.
Example: GPT-2 dimensions and how they scale.
"""

import torch

print("=" * 80)
print("UNDERSTANDING LINEAR LAYER DIMENSIONS IN ATTENTION")
print("=" * 80)

# GPT-2 Small configuration
print("\n1. GPT-2 SMALL MODEL PARAMETERS:")
print("-" * 80)

d_model = 768           # Embedding dimension (fixed for all tokens)
num_heads = 12          # Number of attention heads
seq_length = 1024       # Maximum sequence length
batch_size = 32         # Batch size during training

head_dim = d_model // num_heads  # Per-head dimension

print(f"   d_model (embedding dimension):  {d_model}")
print(f"   num_heads (attention heads):    {num_heads}")
print(f"   head_dim (per-head dimension):  {head_dim} = {d_model} / {num_heads}")
print(f"   batch_size:                     {batch_size}")
print(f"   seq_length:                     {seq_length}")

# Input tensor x
print("\n2. INPUT TENSOR x SHAPE:")
print("-" * 80)
x_shape = (batch_size, seq_length, d_model)
print(f"   x shape: {x_shape}")
print(f"   x shape: (batch_size, seq_length, d_model)")
print(f"   x shape: ({batch_size}, {seq_length}, {d_model})")
print(f"\n   Interpretation:")
print(f"   - {batch_size} sequences in parallel")
print(f"   - Each sequence has {seq_length} tokens")
print(f"   - Each token is a {d_model}-dimensional vector")

# Wq Linear layer
print("\n3. Wq LINEAR LAYER DIMENSIONS:")
print("-" * 80)
print(f"   Wq = Linear(input={d_model}, output={head_dim})")
print(f"   Wq = Linear({d_model}, {head_dim})")
print(f"\n   WHERE DO THESE COME FROM?")
print(f"   • input dimension = d_model = {d_model}")
print(f"     (because x's last dimension is d_model)")
print(f"   • output dimension = head_dim = {head_dim}")
print(f"     (because we want to project to per-head dimension)")

# Wq parameters
wq_weights = d_model * head_dim
wq_biases = head_dim

print(f"\n   Wq PARAMETERS:")
print(f"   • Weights: {wq_weights} = {d_model} × {head_dim}")
print(f"   • Biases:  {wq_biases} = {head_dim}")
print(f"   • Total:   {wq_weights + wq_biases} parameters per head")

# All attention heads
print(f"\n   FOR ALL {num_heads} ATTENTION HEADS:")
print(f"   • Each head has Wq, Wk, Wv (3 linear layers each)")
print(f"   • Each linear layer: {wq_weights} weights + {wq_biases} biases = {wq_weights + wq_biases}")
print(f"   • Total per head: 3 × {wq_weights + wq_biases} = {3 * (wq_weights + wq_biases)} parameters")
print(f"   • All {num_heads} heads: {num_heads} × {3 * (wq_weights + wq_biases)} = {num_heads * 3 * (wq_weights + wq_biases)} parameters")

# Application to batch
print("\n4. APPLYING Wq TO BATCH x:")
print("-" * 80)
print(f"   x shape:      {x_shape}")
print(f"   Wq weight:    ({head_dim}, {d_model})")
print(f"   Wq bias:      ({head_dim},)")
print(f"\n   q = Wq(x)")
q_shape = (batch_size, seq_length, head_dim)
print(f"   q shape:      {q_shape}")
print(f"\n   BROADCASTING: Linear layer applies to LAST dimension")
print(f"   • For each position (batch_idx, seq_idx):")
print(f"     x[batch_idx, seq_idx, :] has shape ({d_model},)")
print(f"     Wq({d_model},) → ({head_dim},)")
print(f"     q[batch_idx, seq_idx, :] has shape ({head_dim},)")

# Different layer example
print("\n" + "=" * 80)
print("PATTERN: How to deduce Linear layer dimensions")
print("=" * 80)
print(f"""
Rule: Linear(input_dim, output_dim)
  • input_dim  = the LAST dimension of the input tensor
  • output_dim = what you want to project to (architectural choice)

Example 1: Query projection in attention
  • Input x shape: (batch, seq_len, d_model)
  • Last dimension: d_model = {d_model}
  • Goal: project to per-head dimension = {head_dim}
  • Wq = Linear({d_model}, {head_dim}) ✓

Example 2: Feed-forward layer
  • Input shape: (batch, seq_len, d_model)
  • Last dimension: d_model = {d_model}
  • Goal: expand to 4×d_model, then back
  • First layer = Linear({d_model}, {d_model * 4})
  • Second layer = Linear({d_model * 4}, {d_model})

Example 3: Output projection
  • Input from all heads concatenated shape: (batch, seq_len, d_model)
  • Last dimension: d_model = {d_model}
  • Goal: project back to d_model (unchanged)
  • Output_proj = Linear({d_model}, {d_model})
""")

# Concrete small example
print("\n" + "=" * 80)
print("CONCRETE EXAMPLE: Tiny GPT-2 setup")
print("=" * 80)

d_model_tiny = 8
num_heads_tiny = 2
head_dim_tiny = d_model_tiny // num_heads_tiny
batch_size_tiny = 2
seq_len_tiny = 3

print(f"\nTiny config: d_model={d_model_tiny}, heads={num_heads_tiny}, head_dim={head_dim_tiny}")

from torch.nn import Linear

Wq = Linear(d_model_tiny, head_dim_tiny)
x_tiny = torch.randn(batch_size_tiny, seq_len_tiny, d_model_tiny)

print(f"\nInput x shape:  {tuple(x_tiny.shape)}")
print(f"Wq:            Linear({d_model_tiny}, {head_dim_tiny})")
print(f"Wq.weight:     {Wq.weight.shape}")
print(f"Wq.bias:       {Wq.bias.shape}")

q_tiny = Wq(x_tiny)
print(f"Output q shape: {tuple(q_tiny.shape)}")
print(f"\nPytorch broadcasts Wq across all positions:")
print(f"  • {batch_size_tiny} × {seq_len_tiny} = {batch_size_tiny * seq_len_tiny} token vectors")
print(f"  • Each {d_model_tiny}D vector → {head_dim_tiny}D vector")
