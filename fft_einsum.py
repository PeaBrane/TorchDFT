import einops
import torch
from torch.backends import opt_einsum

assert opt_einsum.is_available()
opt_einsum.strategy = 'optimal'


TAU = 2 * torch.pi * 1.j


def dft_einsum(input):
    """Applies an order-3 FFT on an input signal, 
    where the first two hierarchies are radix-16.

    Args:
        input (torch.Tensor): the input signal

    Returns:
        torch.Tensor: the DFT'd signal
    """
    
    # reshape input to prepare for hierarchical DFTs
    input_block = einops.rearrange(input, '... (length_1 length_2 length_3) -> ... length_1 length_2 length_3', 
                                   length_1=16, length_2=16)
    last_dim_size = input_block.shape[-1]
    
    # timesteps
    range_16 = torch.arange(16, device=input.device)
    range_small = torch.arange(last_dim_size, device=input.device)
    range_large = torch.arange(16 * last_dim_size, device=input.device)
    
    # DFT matrices
    dft = torch.exp(-(range_16.unsqueeze(-1) * range_16) / 16 * TAU)
    dft_small = torch.exp(-(range_small.unsqueeze(-1) * range_small) / last_dim_size * TAU)
    
    # the twiddle factors
    twid = torch.exp(-(range_16.unsqueeze(-1) * range_large) / (16 ** 2 * last_dim_size) * TAU).reshape(16, 16, -1)
    twid_small = torch.exp(-(range_16.unsqueeze(-1) * range_small) / (16 * last_dim_size) * TAU)
    
    # the actual DFT operations using einsum
    return torch.einsum('...xyz,xf,yg,zh,fyz,gz->...hgf', 
                        input_block.to(torch.cfloat), 
                        dft, dft, dft_small, 
                        twid, twid_small).flatten(-3, -1)


input = torch.rand(2, 4, 1024)
expected = torch.fft.fft(input)
result = dft_einsum(input)

assert torch.allclose(expected, result, rtol=1e-3, atol=1e-3)