"""Check model shape, initialization and gradient coverage without an optimizer."""
import hashlib
import torch
from model import SmallViT

torch.set_num_threads(2)
torch.manual_seed(42)
model = SmallViT()
initial = {n:p.detach().clone() for n,p in model.named_parameters()}
torch.manual_seed(42)
replica = SmallViT()
assert all(torch.equal(initial[n],p) for n,p in replica.named_parameters())
x = torch.randn(2,3,64,64)
logits = model(x)
assert logits.shape == (2,200) and torch.isfinite(logits).all()
torch.nn.functional.cross_entropy(logits,torch.tensor([0,199])).backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
names = [(n,p) for n,p in model.named_parameters()]
hidden = [n for n,p in names if n.startswith('blocks.') and p.ndim == 2]
aux = [n for n,p in names if n not in hidden]
assert len(hidden) == 24 and set(hidden).isdisjoint(aux)
digest = hashlib.sha256()
for n,p in initial.items():
    digest.update(n.encode()); digest.update(p.contiguous().numpy().tobytes())
print('Model, deterministic initialization and backward checks passed.')
print('Parameters:',sum(p.numel() for p in model.parameters()))
print('Initial weights SHA-256:',digest.hexdigest())
print('Muon hidden matrices:',len(hidden),'| auxiliary tensors:',len(aux))
