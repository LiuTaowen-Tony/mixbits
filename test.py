import torch
from block_quant import block_quant

# from torch.utils.cpp_extension import load
# block_quant = load(
#     name='block_quant',
#     sources=['block_quant.cu'],
#     verbose=True
# )

x = torch.randn(10, 10, device='cuda')
y = block_quant(x, 2, 3, 5, 5, False)

print(x)
print(y)
print(y.unique())
