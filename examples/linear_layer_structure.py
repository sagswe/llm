"""
Clarification: Wq is a Linear layer with multiple input and output neurons.
Shows the actual network structure and weight organization.
"""

import torch
from torch.nn import Linear

torch.manual_seed(42)

d_model = 3      # Input neurons
head_dim = 2     # Output neurons

Wq = Linear(d_model, head_dim)

print("=" * 70)
print("LINEAR LAYER STRUCTURE: Linear(input=3, output=2)")
print("=" * 70)

print("\n1. NETWORK ARCHITECTURE:")
print("""
   Input layer (3 neurons)
   ├─ neuron_0
   ├─ neuron_1
   └─ neuron_2
                    ↓ (connections with weights)
   Output layer (2 neurons)
   ├─ neuron_0
   └─ neuron_1
""")

print("\n2. WEIGHT MATRIX (Wq.weight):")
print(f"   Shape: {Wq.weight.shape} = (output_neurons, input_neurons)")
print(f"   This is: (head_dim={head_dim}, d_model={d_model})")
print("\n   Visual representation:")
print(f"            input_0  input_1  input_2")
print(f"   output_0: {Wq.weight[0, 0]:.4f}   {Wq.weight[0, 1]:.4f}   {Wq.weight[0, 2]:.4f}")
print(f"   output_1: {Wq.weight[1, 0]:.4f}   {Wq.weight[1, 1]:.4f}   {Wq.weight[1, 2]:.4f}")

print(f"\n   Total weights: {Wq.weight.numel()} = input_neurons × output_neurons = 3 × 2")

print("\n3. BIAS VECTOR (Wq.bias):")
print(f"   Shape: {Wq.bias.shape} = (output_neurons,)")
print(f"   This is: (head_dim={head_dim},)")
print(f"\n   bias_0 (for output neuron 0): {Wq.bias[0]:.4f}")
print(f"   bias_1 (for output neuron 1): {Wq.bias[1]:.4f}")
print(f"\n   Total biases: {Wq.bias.numel()} = one per output neuron")

print("\n4. COMPUTATION FOR ONE INPUT SAMPLE:")
x_sample = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
print(f"\n   Input: x = {x_sample.tolist()}")
print(f"\n   Output neuron 0:")
print(f"   = weight[0,0]×x[0] + weight[0,1]×x[1] + weight[0,2]×x[2] + bias[0]")
print(f"   = {Wq.weight[0, 0]:.4f}×{x_sample[0]:.1f} + {Wq.weight[0, 1]:.4f}×{x_sample[1]:.1f} + {Wq.weight[0, 2]:.4f}×{x_sample[2]:.1f} + {Wq.bias[0]:.4f}")
result_0 = (Wq.weight[0] * x_sample).sum().item() + Wq.bias[0].item()
print(f"   = {result_0:.4f}")

print(f"\n   Output neuron 1:")
print(f"   = weight[1,0]×x[0] + weight[1,1]×x[1] + weight[1,2]×x[2] + bias[1]")
print(f"   = {Wq.weight[1, 0]:.4f}×{x_sample[0]:.1f} + {Wq.weight[1, 1]:.4f}×{x_sample[1]:.1f} + {Wq.weight[1, 2]:.4f}×{x_sample[2]:.1f} + {Wq.bias[1]:.4f}")
result_1 = (Wq.weight[1] * x_sample).sum().item() + Wq.bias[1].item()
print(f"   = {result_1:.4f}")

output = Wq(x_sample)
print(f"\n   PyTorch output: {output.tolist()}")

print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
print(f"✗ WRONG: 'Wq has 3 weights and 1 bias'")
print(f"✓ CORRECT: 'Wq has {Wq.weight.numel()} weights and {Wq.bias.numel()} biases'")
print(f"\n  Each of the {d_model} input neurons connects to ALL {head_dim} output neurons")
print(f"  → weights = {d_model} × {head_dim} = {d_model * head_dim}")
print(f"  → biases = {head_dim} (one per output neuron)")
