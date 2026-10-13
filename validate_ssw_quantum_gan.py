"""Validate SSW integrity and the encoder-free quantum GAN before training."""
import json
from pathlib import Path
import numpy as np
import torch
from ssw_dataset import validate
from ssw_quantum_gan import QuantumGenerator, Discriminator, prior, step

torch.set_num_threads(2)
torch.manual_seed(123456)
root = Path(__file__).resolve().parent
arrays = dict(np.load(root / "ssw_v1.npz", allow_pickle=False))
checks = validate(arrays)
g, d = QuantumGenerator(), Discriminator()
z = prior(2, torch.Generator().manual_seed(42)).requires_grad_(True)
fake = g(z)
assert fake.shape == (2,8) and torch.isfinite(fake).all()
assert not torch.allclose(fake[0],fake[1])
assert g.quantum.neural_network.num_inputs==4 and g.quantum.neural_network.num_weights==84
assert g.quantum.neural_network.circuit.num_qubits==6
fake.square().mean().backward()
assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
for p in g.parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
g.zero_grad(set_to_none=True)
before = {n:p.detach().clone() for n,p in g.named_parameters()}
go=torch.optim.Adam(g.parameters(),lr=0.001)
do=torch.optim.Adam(d.parameters(),lr=0.001)
losses=step(g,d,go,do,torch.tensor(arrays['train_x'][:2]),z.detach())
for n,p in g.named_parameters():
    assert not torch.equal(before[n],p),n
assert all(p.grad is None and p.requires_grad for p in d.parameters())
state=torch.get_rng_state().clone()
assert torch.equal(g.exact(z.detach()),g.exact(z.detach()))
assert torch.equal(state,torch.get_rng_state())
report={'status':'passed','dataset':checks,'input_shape':[2,4],'output_shape':[2,8],
        'encoder':False,'quantum_weights':84,'losses':losses,
        'checks':['latent sensitivity','finite input/quantum/decoder gradients','generator updates',
                  'D frozen during G update','deterministic noiseless evaluation']}
(root/'ssw_gan_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
