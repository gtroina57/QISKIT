"""Create an isolated generator learning-rate experiment notebook."""
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parent
nb = nbformat.v4.new_notebook()
nb.metadata = nbformat.read(root/'GAN_03_SSW_quantum_generator.ipynb', as_version=4).metadata
nb.cells = [nbformat.v4.new_markdown_cell('''# SSW: lower generator learning rate

Continue the shared step-12 checkpoint to step 64 with G learning rate
0.0002 instead of 0.001. D retains learning rate 0.001 and one update per G
update. Architecture, batch size four, dataset noise and evaluation inputs
are unchanged. Compare with Notebook 04. This is one stochastic continuation;
unseeded quantum sampling prevents matched measurement draws.

Quantum and decoder gradient L2 norms are recorded before each G update.
Their magnitudes alone do not establish whether quantum features are useful.
Best checkpoint selection uses validation MMD only; also inspect diversity,
feature spread and logical residual. The selected best can remain step 12.
'''), nbformat.v4.new_code_cell('''import json
from pathlib import Path
import numpy as np
import torch
from ssw_quantum_gan import train_run
ROOT = Path.cwd()
torch.set_num_threads(2)
arrays = dict(np.load(ROOT/'ssw_v1.npz', allow_pickle=False))
OUTPUT_DIR = ROOT/'ssw_gan_lower_lr_results'
report = train_run(arrays, OUTPUT_DIR, steps=64, batch_size=4,
    resume=ROOT/'ssw_gan_results/final.pt', discriminator_updates=1,
    generator_lr=0.0002)
assert report['generator_learning_rate'] == 0.0002
assert report['discriminator_learning_rate'] == 0.001
state = torch.load(OUTPUT_DIR/'final.pt', weights_only=True)
assert state['generator_optimizer']['param_groups'][0]['lr'] == 0.0002
assert all(int(s['step']) == 64 for s in state['generator_optimizer']['state'].values())
assert all(int(s['step']) == 64 for s in state['discriminator_optimizer']['state'].values())
assert all(np.isfinite(r[k]) for r in report['losses'] if r['step'] > 12
           for k in ('quantum_gradient_norm', 'decoder_gradient_norm'))
print('Checkpoint rates, optimizer counters and gradient diagnostics verified.')
'''), nbformat.v4.new_code_cell('''import matplotlib.pyplot as plt
control = json.loads((ROOT/'ssw_gan_longer_results/report.json').read_text())
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
for data, label in [(control, 'G lr 0.001'), (report, 'G lr 0.0002 after step 12')]:
    for ax, key in zip(axes, ['rbf_mmd_squared', 'feature_std_mae', 'logic_residual_mse']):
        ax.plot([r['step'] for r in data['history']], [r[key] for r in data['history']], label=label)
        ax.set_title(key); ax.set_xlabel('Generator updates'); ax.grid(alpha=.3)
axes[0].legend()
fig.savefig(OUTPUT_DIR/'control_comparison.png', dpi=150)
plt.show()
rows = [r for r in report['losses'] if r['step'] > 12]
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
for key in ['quantum_gradient_norm', 'decoder_gradient_norm']:
    ax.plot([r['step'] for r in rows], [r[key] for r in rows], label=key)
ax.set_yscale('symlog', linthresh=1e-8)
ax.set_xlabel('Generator updates'); ax.set_ylabel('Gradient L2 norm'); ax.legend()
fig.savefig(OUTPUT_DIR/'gradients.png', dpi=150)
plt.show()
print('Best step:', report['best_step'])
print('Final diagnostics:', json.dumps(report['history'][-1], indent=2))
''')]
nbformat.write(nb, root/'GAN_06_SSW_lower_generator_lr.ipynb')
print('Created Notebook 06.')
