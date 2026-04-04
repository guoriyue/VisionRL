"""Fused activation kernels: SwiGLU, GeGLU, ReLU².

SwiGLU(gate, up) = SiLU(gate) * up = (gate * sigmoid(gate)) * up

Why fuse? Without fusion:
  1. Kernel launch: silu(gate)        — read gate, write silu_out
  2. Kernel launch: silu_out * up     — read silu_out + up, write output
  = 2 launches, 4 global memory reads, 2 writes

Fused:
  1. Single kernel: read gate + up, write output
  = 1 launch, 2 reads, 1 write (3x less memory traffic)
"""

import triton
import triton.language as tl


@triton.jit
def swiglu_fwd_kernel(
    # Inputs
    gate_ptr,       # [total_tokens, intermediate_dim]
    up_ptr,         # [total_tokens, intermediate_dim]
    # Output
    output_ptr,     # [total_tokens, intermediate_dim]
    # Shape
    num_elements,   # total_tokens * intermediate_dim
    # Tuning
    BLOCK: tl.constexpr = 1024,
):
    """SwiGLU forward: output = SiLU(gate) * up
    
    SiLU(x) = x * sigmoid(x)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elements
    
    gate = tl.load(gate_ptr + offsets, mask=mask)
    up = tl.load(up_ptr + offsets, mask=mask)

    # SiLU = x * sigmoid(x) — sigmoid requires fp32
    gate_f32 = gate.to(tl.float32)
    sigmoid_gate = tl.sigmoid(gate_f32)
    silu_gate = gate_f32 * sigmoid_gate

    output = silu_gate * up.to(tl.float32)
    tl.store(output_ptr + offsets, output.to(output_ptr.dtype.element_ty), mask=mask)


@triton.jit
def swiglu_bwd_kernel(
    # Forward inputs (saved for backward)
    gate_ptr,       # [total_tokens, intermediate_dim]
    up_ptr,         # [total_tokens, intermediate_dim]
    # Upstream gradient
    dout_ptr,       # [total_tokens, intermediate_dim]
    # Output gradients
    dgate_ptr,      # [total_tokens, intermediate_dim]
    dup_ptr,        # [total_tokens, intermediate_dim]
    # Shape
    num_elements,
    BLOCK: tl.constexpr = 1024,
):
    """SwiGLU backward.
    
    Forward: out = SiLU(gate) * up = gate * sigmoid(gate) * up
    
    d(out)/d(up)   = SiLU(gate) = gate * sigmoid(gate)
    d(out)/d(gate) = up * d(SiLU(gate))/d(gate)
                   = up * (sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate)))
                   = up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elements
    
    gate = tl.load(gate_ptr + offsets, mask=mask)
    up = tl.load(up_ptr + offsets, mask=mask)
    dout = tl.load(dout_ptr + offsets, mask=mask)

    # Compute in fp32 for sigmoid compatibility
    gate_f32 = gate.to(tl.float32)
    up_f32 = up.to(tl.float32)
    dout_f32 = dout.to(tl.float32)

    sigmoid_gate = tl.sigmoid(gate_f32)
    silu_gate = gate_f32 * sigmoid_gate

    # d(out)/d(up) = silu(gate)
    dup = dout_f32 * silu_gate

    # d(out)/d(gate) = up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    dsilu = sigmoid_gate * (1.0 + gate_f32 * (1.0 - sigmoid_gate))
    dgate = dout_f32 * up_f32 * dsilu

    out_dtype = gate_ptr.dtype.element_ty
    tl.store(dgate_ptr + offsets, dgate.to(out_dtype), mask=mask)
    tl.store(dup_ptr + offsets, dup.to(out_dtype), mask=mask)


@triton.jit
def geglu_fwd_kernel(
    gate_ptr, up_ptr, output_ptr,
    num_elements,
    BLOCK: tl.constexpr = 1024,
):
    """GeGLU forward: output = GELU(gate) * up"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elements
    
    gate = tl.load(gate_ptr + offsets, mask=mask)
    up = tl.load(up_ptr + offsets, mask=mask)

    # GELU approximation in fp32 (tanh requires fp32)
    gate_f32 = gate.to(tl.float32)
    SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
    gate3 = gate_f32 * gate_f32 * gate_f32
    inner = SQRT_2_OVER_PI * (gate_f32 + 0.044715 * gate3)
    gelu_gate = 0.5 * gate_f32 * (1.0 + tl.math.tanh(inner))

    output = gelu_gate * up.to(tl.float32)
    tl.store(output_ptr + offsets, output.to(output_ptr.dtype.element_ty), mask=mask)


@triton.jit
def swiglu_fused_gate_up_kernel(
    # Input: contiguous [top_k, 2*I] with gate in [:, :I] and up in [:, I:]
    gate_up_ptr,
    # Output: [top_k, I] — SiLU(gate) * up
    output_ptr,
    # half_N = I (the intermediate_dim)
    half_N,
    # total_rows = top_k
    total_rows,
    BLOCK: tl.constexpr = 1024,
):
    """SwiGLU from fused gate_up bmm output.

    Reads gate_up[row, col] and gate_up[row, col + half_N] as gate/up pairs,
    computes SiLU(gate) * up, writes to output[row, col].
    Processes element-wise over (total_rows * half_N) elements.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    num_elements = total_rows * half_N
    mask = offsets < num_elements

    # Map flat offset to (row, col) in the [total_rows, half_N] output
    row = offsets // half_N
    col = offsets % half_N

    # Load gate from first half, up from second half of the 2*I dimension
    gate_offset = row * (2 * half_N) + col
    up_offset = gate_offset + half_N

    gate = tl.load(gate_up_ptr + gate_offset, mask=mask)
    up = tl.load(gate_up_ptr + up_offset, mask=mask)

    # SiLU = x * sigmoid(x) — sigmoid requires fp32
    gate_f32 = gate.to(tl.float32)
    sigmoid_gate = tl.sigmoid(gate_f32)
    silu_gate = gate_f32 * sigmoid_gate

    output = silu_gate * up.to(tl.float32)
    tl.store(output_ptr + offsets, output.to(output_ptr.dtype.element_ty), mask=mask)


@triton.jit
def relu2_fwd_kernel(
    gate_ptr, up_ptr, output_ptr,
    num_elements,
    BLOCK: tl.constexpr = 1024,
):
    """ReLU² GLU: output = ReLU(gate)² * up"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elements
    
    gate = tl.load(gate_ptr + offsets, mask=mask)
    up = tl.load(up_ptr + offsets, mask=mask)
    
    relu_gate = tl.maximum(gate, 0.0)
    output = relu_gate * relu_gate * up
    tl.store(output_ptr + offsets, output, mask=mask)
