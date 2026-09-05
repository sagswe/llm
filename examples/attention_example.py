"""
Concrete example of Wq projection with small values.
Shows the actual matrix multiplication step-by-step.
"""

import torch
from torch.nn import Linear

# Set seed for reproducibility
torch.manual_seed(42)

# Small dimensions for clarity
d_model = 3      # Model width
head_dim = 2     # Attention head width
batch_size = 1
seq_length = 2

print("=" * 60)
print("EXAMPLE: Query Projection (Wq)")
print("=" * 60)

# Create a Wq layer
Wq = Linear(d_model, head_dim)

# Print the weight matrix and bias
print("\n1. Wq Weight Matrix (shape: head_dim x d_model):")
print(f"   Shape: {Wq.weight.shape}")
print(Wq.weight)

print("\n2. Wq Bias Vector (shape: head_dim):")
print(f"   Shape: {Wq.bias.shape}")
print(Wq.bias)

# Create a small input tensor: batch_size=1, seq_length=2, d_model=3
# Two tokens, each with 3-dimensional embeddings
x = torch.tensor([
    [[0.1, 0.2, 0.3],   # Token 1
     [0.4, 0.5, 0.6]]   # Token 2
], dtype=torch.float32)

print("\n3. Input Tensor x (shape: batch_size x seq_length x d_model):")
print(f"   Shape: {x.shape}")
print(x)

# Manual calculation for x[0, 0] (first token)
print("\n4. MANUAL CALCULATION - x[0, 0]:")
x_0_0 = x[0, 0]  # [0.1, 0.2, 0.3]
print(f"   x[0, 0] = {x_0_0}")
print(f"   q[0, 0] = Wq(x[0, 0]) = Wq.weight @ x[0, 0] + Wq.bias")

# For each head_dim output
for i in range(head_dim):
    w = Wq.weight[i]  # Weight row for this output dimension
    b = Wq.bias[i]    # Bias for this output dimension
    result = (w * x_0_0).sum().item() + b.item()
    print(f"   q[0, 0, {i}] = {w} · {x_0_0} + {b:.4f}")
    print(f"             = {(w * x_0_0).sum().item():.6f} + {b.item():.6f} = {result:.6f}")

print("\n5. MANUAL CALCULATION - x[0, 1] (second token):")
x_0_1 = x[0, 1]  # [0.4, 0.5, 0.6]
print(f"   x[0, 1] = {x_0_1}")
print(f"   q[0, 1] = Wq(x[0, 1]) = Wq.weight @ x[0, 1] + Wq.bias")
for i in range(head_dim):
    w = Wq.weight[i]
    b = Wq.bias[i]
    result = (w * x_0_1).sum().item() + b.item()
    print(f"   q[0, 1, {i}] = {w} · {x_0_1} + {b:.4f}")
    print(f"             = {(w * x_0_1).sum().item():.6f} + {b.item():.6f} = {result:.6f}")

# Apply Wq using PyTorch
q = Wq(x)
print("\n6. Result of Wq(x) using PyTorch:")
print(f"   Shape: {q.shape} (batch_size x seq_length x head_dim)")
print(q)

print("\n7. VERIFICATION:")
print(f"   q[0, 0] = {q[0, 0]}")
print(f"   q[0, 1] = {q[0, 1]}")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print(f"• x shape: {x.shape} (batch_size=1, sequence_length=2, d_model=3)")
print(f"• Wq transforms each x[batch, position, :] from d_model=3 → head_dim=2")
print(f"• q = Wq(x) applies the transformation to all tokens")
print(f"• q[0, 0] is the query for position 0 (transformed x[0, 0])")
print(f"• q[0, 1] is the query for position 1 (transformed x[0, 1])")
print(f"• Output q shape: {q.shape}")
