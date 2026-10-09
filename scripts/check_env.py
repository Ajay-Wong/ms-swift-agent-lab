import importlib.metadata as metadata
import torch

for package in ('ms-swift', 'torch', 'transformers', 'peft', 'trl', 'vllm'):
    print(f'{package}: {metadata.version(package)}', flush=True)
assert torch.cuda.is_available(), 'CUDA unavailable'
print('GPU:', torch.cuda.get_device_name(0))
print('CUDA:', torch.version.cuda, 'capability:', torch.cuda.get_device_capability(0))
x = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
(x @ x.T).float().square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all()
print('BF16 GPU forward/backward: PASS')
