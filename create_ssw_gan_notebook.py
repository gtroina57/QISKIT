"""Create a separate encoder-free SSW quantum GAN experiment notebook."""
from pathlib import Path
import nbformat

root=Path(__file__).resolve().parent
n=nbformat.v4.new_notebook()
n.metadata['kernelspec']={'display_name':'DOUBT_QC','language':'python','name':'python3'}
n.cells=[
nbformat.v4.new_markdown_cell('''# SSW v1.0 — quantum generator GAN, without encoder

`z (4 random angles) → six-qubit SamplerQNN → 64 probabilities → learned decoder → x (8 features)`.
The classical discriminator sees only eight-dimensional observations. No encoder,
latent-pair discriminator, factor supervision, reconstruction loss or BiGAN objective.

The quantum circuit retains the Gaussian experiment's Hadamards, four Ry inputs,
six-repetition efficient_su2 ansatz, 84 quantum weights, Qiskit Machine Learning
and TorchConnector. The new decoder is 64→32→8 with a linear output. SSW noise
is unbounded, so there is no sigmoid, clipping or normalization of observations.
The decoder receives 64 times the probability vector as a feature scaling.
Discriminator: 8→20→1 logits; non-saturating GAN BCE, separate Adam optimizers
(lr=0.001, betas=(0.7,0.999)). The quantum sampler is unseeded, with 1024 shots.

The supplied SSW equations supersede the earlier 8×8 JEPA images. Dataset files
were recreated from that specification because the original companion code was
not found in the workspace. They are deterministic for the documented integer
selection and draw order; byte equality to an unavailable reference is not claimed.
'''),
nbformat.v4.new_code_cell('''import json
import sys
import subprocess
from pathlib import Path
import numpy as np
import torch

ROOT=Path.cwd()
assert (ROOT/'ssw_quantum_gan.py').is_file()
torch.set_num_threads(2)
from ssw_dataset import validate
from ssw_quantum_gan import train_run
arrays=dict(np.load(ROOT/'ssw_v1.npz',allow_pickle=False))
print(validate(arrays))
for name in ['train','val_iid','test_iid','test_ood']:
    print(name,arrays[name+'_x'].shape)
print(json.loads((ROOT/'ssw_v1_metadata.json').read_text()))'''),
nbformat.v4.new_markdown_cell('''## Verify before training
Check dataset equations and OOD separation, latent sensitivity, gradients,
optimizer updates and deterministic evaluation. The next cell stops on failure.'''),
nbformat.v4.new_code_cell('''completed=subprocess.run([sys.executable,str(ROOT/'validate_ssw_quantum_gan.py')],
                         cwd=ROOT,capture_output=True,text=True)
print(completed.stdout)
if completed.stderr: print(completed.stderr)
completed.check_returncode()'''),
nbformat.v4.new_markdown_cell('''## Short integration baseline

Twelve optimizer steps, batch size four, sample randomly from the complete
2,000-observation training pool. This processes 48 training examples, not a full
epoch or a convergence experiment. Increase STEPS only after inspecting results.

Every four steps, evaluate the same 400 latent vectors with exact probabilities
and the learned decoder. Select checkpoints by val_iid MMD², with kernel bandwidth
determined from training data only. Test splits never select weights. The OOD
comparison is exploratory: an unconditional GAN trained on 12 configurations is
not trained to generate the four excluded configurations. No OOD success is assumed.
'''),
nbformat.v4.new_code_cell('''STEPS=12
BATCH_SIZE=4
OUTPUT_DIR=ROOT/'ssw_gan_results'
report=train_run(arrays,OUTPUT_DIR,steps=STEPS,batch_size=BATCH_SIZE)
print(json.dumps(report,indent=2))'''),
nbformat.v4.new_code_cell('''import matplotlib.pyplot as plt
history=report['history']
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
axes[0].plot([r['step'] for r in report['losses']],[r['generator_loss'] for r in report['losses']],label='G')
axes[0].plot([r['step'] for r in report['losses']],[r['discriminator_loss'] for r in report['losses']],label='D')
axes[0].set_ylabel('Adversarial loss'); axes[0].legend()
axes[1].plot([r['step'] for r in history],[r['rbf_mmd_squared'] for r in history],marker='o')
axes[1].set_ylabel('Fixed noiseless validation MMD²')
for ax in axes: ax.set_xlabel('Training step'); ax.grid(alpha=0.3)
fig.savefig(OUTPUT_DIR/'training.png',dpi=150,bbox_inches='tight')
plt.show()

examples=torch.load(OUTPUT_DIR/'examples.pt',weights_only=True)
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
vmin=min(examples['validation_real'].min().item(),examples['selected_generated'].min().item())
vmax=max(examples['validation_real'].max().item(),examples['selected_generated'].max().item())
for ax,key,title in zip(axes,['validation_real','selected_generated'],['Real SSW vectors','Generated SSW vectors']):
    image=ax.imshow(examples[key][:24].numpy(),aspect='auto',vmin=vmin,vmax=vmax,cmap='viridis')
    ax.set_title(title); ax.set_xlabel('Feature'); ax.set_ylabel('Independent sample')
    ax.set_xticks(range(8),[f'x{i+1}' for i in range(8)])
fig.colorbar(image,ax=list(axes),label='Feature value',shrink=0.8)
fig.savefig(OUTPUT_DIR/'samples.png',dpi=150,bbox_inches='tight')
plt.show()'''),
nbformat.v4.new_markdown_cell('''## Interpretation

These heatmaps show eight-feature vectors, not images, and displayed real and
generated samples are not paired. Configuration counts threshold the first four
generated features; the logical residual compares all eight features with the
corresponding clean world. This is a coarse diagnostic, not known generated labels.
Compare variance, coverage, logic and MMD together; low variance can indicate collapse.
Best/latest/final weights and optimizer states are saved in `ssw_gan_results`.
Unseeded measurement draws cannot be replayed exactly from saved RNG states.
The best checkpoint is checked by reloading and recomputing its validation score.
''')]
nbformat.write(n,root/'GAN_03_SSW_quantum_generator.ipynb')
print('Created GAN_03_SSW_quantum_generator.ipynb')
