"""Test the D-update ratio and prepare its isolated experiment notebook."""
from pathlib import Path
import torch
import nbformat
from ssw_quantum_gan import QuantumGenerator,Discriminator,prior,step
import numpy as np

root=Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.manual_seed(123456)
g,d=QuantumGenerator(),Discriminator()
go=torch.optim.Adam(g.parameters(),lr=0.001,betas=(0.7,0.999))
do=torch.optim.Adam(d.parameters(),lr=0.001,betas=(0.7,0.999))
real=torch.tensor(np.load(root/'ssw_v1.npz')['train_x'][:2])
loss=step(g,d,go,do,real,prior(2,torch.Generator().manual_seed(1)),3)
assert all(s['step'].item()==1 for s in go.state.values())
assert all(s['step'].item()==3 for s in do.state.values())
assert all(torch.isfinite(p.grad).all() for p in g.parameters())
assert all(p.grad is None and p.requires_grad for p in d.parameters())
print('Update-ratio check passed: D=3, G=1; finite G gradients; D frozen for G.',flush=True)
n=nbformat.read(root/'GAN_03_SSW_quantum_generator.ipynb',as_version=4)
n.cells.insert(0,nbformat.v4.new_markdown_cell('''# SSW stabilization test: three D updates per G update

Resume from the original step-12 checkpoint to step 64. The **only training
change** is three discriminator optimizer updates rather than one, using the
same real and generated minibatch for those three updates. Fake observations
are generated once for D and reused, so extra D updates consume no additional
latent or quantum samples. G still receives one update with a fresh forward pass.

Generator, decoder, discriminator architecture, learning rates, batch size four,
SSW observation noise and fixed evaluation inputs are unchanged. No encoder.
Compare with `GAN_04_SSW_quantum_generator_longer.ipynb`. Quantum sampling is
unseeded, so this is a controlled setting comparison, not bitwise matched draws
or a multi-seed conclusion. Results are saved in `ssw_gan_d3_results`.
'''))
for c in n.cells:
    if c.cell_type=='code':
        if 'report=train_run' in c.source:
            c.source=c.source.replace('STEPS=12','STEPS=64').replace("ROOT/'ssw_gan_results'","ROOT/'ssw_gan_d3_results'")
            c.source=c.source.replace('batch_size=BATCH_SIZE)',"batch_size=BATCH_SIZE, resume=ROOT/'ssw_gan_results/final.pt', discriminator_updates=3)")
        c.execution_count=None
        c.outputs=[]
    else:
        c.source=c.source.replace('Twelve optimizer steps','Sixty-four total generator optimizer steps').replace('48 training examples','256 training examples')
        c.source=c.source.replace('saved in `ssw_gan_results`','saved in `ssw_gan_d3_results`')
nbformat.write(n,root/'GAN_05_SSW_discriminator_updates.ipynb')
