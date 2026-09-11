# Generated from our custom IR by compiler.generate_triton.
# Operation: %relu = fused_affine_relu(x, scale, bias)
import triton
import triton.language as tl


@triton.jit
def affine_relu_kernel(x_ptr, scale_ptr, bias_ptr, out_ptr, n_elements,
                       BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    affine = x * scale + bias
    # Preserve NaNs, as torch.relu does.
    result = tl.where(affine < 0.0, 0.0, affine)
    tl.store(out_ptr + offsets, result, mask=mask)
