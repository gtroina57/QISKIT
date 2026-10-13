"""Create a separate longer run and verify resume logic without new updates."""
from pathlib import Path
import json
import numpy as np
import torch
import nbformat
from ssw_quantum_gan import train_run

root=Path(__file__).resolve().parent
torch.set_num_threads(2)
arrays=dict(np.load(root/'ssw_v1.npz',allow_pickle=False))
report=train_run(arrays,root/'ssw_resume_validation',steps=12,
                 resume=root/'ssw_gan_results/final.pt')
assert report['new_steps']==0 and report['steps']==12
previous=json.loads((root/'ssw_gan_results/report.json').read_text())
assert report['history']==previous['history'] and report['losses']==previous['losses']
print('Resume test passed: unchanged metrics and history, zero new updates.')
n=nbformat.read(root/'GAN_03_SSW_quantum_generator.ipynb',as_version=4)
n.cells.insert(0,nbformat.v4.new_markdown_cell('''# Longer SSW quantum GAN run — continuation

Resume the saved 12-step model and both optimizer states, keeping the architecture,
four-dimensional prior, dataset observation noise (std 0.05), learning rates,
minibatch size, and fixed evaluation inputs unchanged. No encoder is included.
This run reaches **64 total optimizer steps** (52 additional), processing 256
training examples in total from the 2,000-sample pool. It is still not a full
training epoch. Save results separately in `ssw_gan_longer_results`.

The original 12-step experiment is preserved. Quantum measurement sampling remains
unseeded, so restoration does not reproduce the original sequence of measurement draws.
'''))
for c in n.cells:
    if c.cell_type=='code':
        if 'report=train_run' in c.source:
            c.source=c.source.replace('STEPS=12','STEPS=64').replace("ROOT/'ssw_gan_results'","ROOT/'ssw_gan_longer_results'")
            c.source=c.source.replace('batch_size=BATCH_SIZE)',"batch_size=BATCH_SIZE, resume=ROOT/'ssw_gan_results/final.pt')")
        c.execution_count=None
        c.outputs=[]
    elif 'Twelve optimizer steps' in c.source:
        c.source=c.source.replace('Twelve optimizer steps','Sixty-four total optimizer steps').replace('48 training examples','256 training examples')
    if c.cell_type=='markdown':
        c.source=c.source.replace('saved in `ssw_gan_results`','saved in `ssw_gan_longer_results`')
nbformat.write(n,root/'GAN_04_SSW_quantum_generator_longer.ipynb')
print('Created GAN_04_SSW_quantum_generator_longer.ipynb')
