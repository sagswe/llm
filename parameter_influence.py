"""
Which parameters influence Wq, Wk, Wv dimensions?
From (batch, seq_len, d_model, head_dim), what matters?
"""

import torch
from torch.nn import Linear

print("=" * 80)
print("PARAMETER INFLUENCE ON LINEAR LAYER DIMENSIONS")
print("=" * 80)

# All available parameters
batch = 32
seq_len = 1024
d_model = 768
head_dim = 64

print("\nGIVEN PARAMETERS:")
print(f"  batch     = {batch}     (batch size)")
print(f"  seq_len   = {seq_len}   (sequence length)")
print(f"  d_model   = {d_model}   (embedding dimension)")
print(f"  head_dim  = {head_dim}  (per-head dimension)")

print("\n" + "=" * 80)
print("LINEAR LAYER DIMENSIONS FOR Wq, Wk, Wv")
print("=" * 80)

# What affects input dimension?
print("\n1. INPUT DIMENSION (Input Neurons):")
print("-" * 80)
print(f"   Input tensor x shape: (batch, seq_len, d_model)")
print(f"   Input tensor x shape: ({batch}, {seq_len}, {d_model})")
print(f"\n   ✓ INFLUENCED BY: d_model")
print(f"   ✗ NOT influenced by: batch, seq_len, head_dim")
print(f"\n   WHY? Linear layer applies to the LAST dimension")
print(f"   Linear(..., input=d_model, ...)")

# What affects output dimension?
print("\n2. OUTPUT DIMENSION (Output Neurons):")
print("-" * 80)
print(f"   Design choice: Project to per-head dimension")
print(f"   Output per head: head_dim = {head_dim}")
print(f"\n   ✓ INFLUENCED BY: head_dim")
print(f"   ✗ NOT influenced by: batch, seq_len, d_model (directly)")
print(f"\n   WHY? Architectural choice to split d_model across heads")
print(f"   Linear(..., output=head_dim)")

# Complete picture
print("\n" + "=" * 80)
print("WOUNDING: Linear(input_dim, output_dim)")
print("=" * 80)

print(f"""
Input dimension  = d_model   = {d_model}
Output dimension = head_dim  = {head_dim}

Therefore:
  Wq = Linear(d_model={d_model}, head_dim={head_dim})
  Wk = Linear(d_model={d_model}, head_dim={head_dim})
  Wv = Linear(d_model={d_model}, head_dim={head_dim})
""")

print("=" * 80)
print("BATCH AND SEQ_LEN DO NOT AFFECT LINEAR LAYER STRUCTURE")
print("=" * 80)

# Show with concrete examples
print("\nExample 1: Change batch size")
print("-" * 80)
batch_old = 32
batch_new = 64
print(f"  Old: batch={batch_old}, seq_len={seq_len}")
print(f"  New: batch={batch_new}, seq_len={seq_len}")
print(f"  Linear layer changes? NO ✓")
print(f"  Still: Linear({d_model}, {head_dim})")

print("\nExample 2: Change seq_len")
print("-" * 80)
seq_old = 1024
seq_new = 2048
print(f"  Old: batch={batch}, seq_len={seq_old}")
print(f"  New: batch={batch}, seq_len={seq_new}")
print(f"  Linear layer changes? NO ✓")
print(f"  Still: Linear({d_model}, {head_dim})")

print("\nExample 3: Change d_model")
print("-" * 80)
d_model_old = 768
d_model_new = 1024
print(f"  Old: d_model={d_model_old}")
print(f"  New: d_model={d_model_new}")
print(f"  Linear layer changes? YES ✓")
print(f"  Old: Linear({d_model_old}, {head_dim})")
print(f"  New: Linear({d_model_new}, {head_dim})")

print("\nExample 4: Change head_dim")
print("-" * 80)
head_old = 64
head_new = 128
print(f"  Old: head_dim={head_old}")
print(f"  New: head_dim={head_new}")
print(f"  Linear layer changes? YES ✓")
print(f"  Old: Linear({d_model}, {head_old})")
print(f"  New: Linear({d_model}, {head_new})")

print("\n" + "=" * 80)
print("SUMMARY: INPUT AND OUTPUT NEURONS")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────┐
│ NUMBER OF INPUT NEURONS  = d_model = {d_model}      │
│ NUMBER OF OUTPUT NEURONS = head_dim = {head_dim}    │
└─────────────────────────────────────────────────────┘

batch and seq_len:
  • Do NOT determine Linear layer structure
  • Are just batch dimensions that get broadcast
  • Allow processing multiple sequences in parallel
  • But the transformation per token is the same

d_model and head_dim:
  • DO determine Linear layer structure
  • Are fixed for the model architecture
  • Cannot change per-batch or per-position
""")

# Verification with actual tensors
print("\n" + "=" * 80)
print("VERIFICATION: Applying to different batch/seq_len")
print("=" * 80)

Wq = Linear(d_model, head_dim)
print(f"\nCreated: Wq = Linear({d_model}, {head_dim})")
print(f"Wq.weight shape: {Wq.weight.shape} = (output={head_dim}, input={d_model})")
print(f"Wq.bias shape:   {Wq.bias.shape} = (output={head_dim},)")

print("\nApply to different batch sizes (same Wq):")
for test_batch in [1, 32, 64]:
    x = torch.randn(test_batch, seq_len, d_model)
    q = Wq(x)
    print(f"  Input:  ({test_batch}, {seq_len}, {d_model}) → Output: {tuple(q.shape)}")
    print(f"    ✓ Wq unchanged, just broadcasts across batch")

print("\nApply to different seq_len (same Wq):")
for test_seq in [256, 512, 1024, 2048]:
    x = torch.randn(batch, test_seq, d_model)
    q = Wq(x)
    print(f"  Input:  ({batch}, {test_seq}, {d_model}) → Output: {tuple(q.shape)}")
    print(f"    ✓ Wq unchanged, just broadcasts across sequence")
