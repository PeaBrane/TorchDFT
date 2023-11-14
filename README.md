This repository is dedicated to exploring and implementing the Discrete Fourier Transform (DFT) in various ways using PyTorch.
The current goal is mostly educational, therefore the implementations may not always be the most optimal, 
but they are designed to provide clear insights into the workings of DFT algorithms.

In the near future, there are plans to optimize these implementations to interact better with kernel fusion and compilation
to provide actual speed ups when implementing DFTs on GPUs, in particular with `torch.compile` or `triton`