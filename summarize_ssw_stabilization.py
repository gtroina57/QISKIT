"""Validate and document the completed discriminator-update experiment."""
from pathlib import Path
import json
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
out = root / 'ssw_gan_d3_results'
baseline = json.loads((root/'ssw_gan_results/report.json').read_text())
control = json.loads((root/'ssw_gan_longer_results/report.json').read_text())
trial = json.loads((out/'report.json').read_text())
assert trial['history'][:len(baseline['history'])] == baseline['history']
assert trial['losses'][:12] == baseline['losses']
assert trial['steps'] == 64 and trial['discriminator_updates_per_generator'] == 3
state = torch.load(out/'final.pt', weights_only=True)
for key, expected in [('generator_optimizer',64),('discriminator_optimizer',168)]:
    assert all(int(s['step']) == expected for s in state[key]['state'].values())
nbpath = root/'GAN_05_SSW_discriminator_updates.ipynb'
nb = json.loads(nbpath.read_text(encoding='utf-8'))
code = [c for c in nb['cells'] if c['cell_type']=='code']
assert all(c['execution_count'] is not None for c in code)
assert not any(o['output_type']=='error' for c in code for o in c['outputs'])
fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
for report,label in [(control,'One D update'),(trial,'Three D updates after step 12')]:
    for ax,key,title in zip(axes,['rbf_mmd_squared','feature_std_mae','logic_residual_mse'],
                             ['Validation MMD² (lower is better)','Feature standard-deviation error','Structured-feature residual MSE']):
        ax.plot([h['step'] for h in report['history']],[h[key] for h in report['history']],label=label)
        ax.set_title(title); ax.set_xlabel('Generator updates'); ax.grid(alpha=.3)
axes[0].legend()
fig.savefig(out/'control_comparison.png',dpi=150)
plt.close(fig)
rows=[]
for label,h in [('Shared starting model',baseline['history'][-1]),('One D update: final',control['history'][-1]),('Three D updates: final',trial['history'][-1])]:
    rows.append(f"| {label} | {h['rbf_mmd_squared']:.4f} | {h['feature_mean_mae']:.4f} | {h['feature_std_mae']:.4f} | {h['covered_configurations']} | {h['logic_residual_mse']:.4f} |")
text = '''# SSW discriminator-update experiment

Executed `GAN_05_SSW_discriminator_updates.ipynb`, continuing the saved step-12
model through step 64. Only the discriminator update ratio changed: three
updates reuse the same real/generated minibatch before each generator update.
The quantum generator, decoder, latent prior, learning rates, batch size four,
SSW v1.0 observations and evaluation inputs remain unchanged. No encoder was added.
No hardware noise model was added; the dataset retains its specified observation noise.

| Checkpoint | Validation MMD² | Mean MAE | Std MAE | Thresholded pattern coverage | Logic residual MSE |
|---|---:|---:|---:|---:|---:|
'''+'\n'.join(rows)+f'''

Selected best checkpoint: step **{trial['best_step']}**. Selection uses validation
MMD² only. Pattern assignments threshold the first four generated features;
they do not establish that the generated vectors satisfy the world's equations.
The residual is an additional diagnostic, not the training objective.

Validated: inherited history/losses match the original run, all notebook code
cells executed without errors, and saved optimizer counters record 64 generator
updates and 168 discriminator updates (12 initial + 52 × 3). Training used
256 real-example draws with replacement, not a complete training epoch.

This is one stochastic continuation. Unseeded quantum sampling prevents exact
replay and the comparison does not establish an effect across random seeds.
Review both final and best checkpoints; the notebook's initial sample figure
uses the selected best checkpoint.

![Control comparison](ssw_gan_d3_results/control_comparison.png)

![Best versus final samples](ssw_gan_d3_results/best_vs_final.png)

Finite-shot diagnostic: `ssw_gan_d3_results/sampled_evaluation.json`.
'''
(root/'SSW_STABILIZATION.md').write_text(text,encoding='utf-8')
nb['cells'].append({'cell_type':'markdown','metadata':{},'source':[
    '## Completed experiment\n',
    f"Best validation checkpoint: step {trial['best_step']}. Final MMD²: {trial['history'][-1]['rbf_mmd_squared']:.4f}, versus {control['history'][-1]['rbf_mmd_squared']:.4f} for the one-update control.\n\n",
    'See [the experiment report](SSW_STABILIZATION.md) for quality diagnostics and limitations.\n\n',
    '![Control comparison](ssw_gan_d3_results/control_comparison.png)\n\n',
    '![Best and final checkpoints](ssw_gan_d3_results/best_vs_final.png)\n']})
nbpath.write_text(json.dumps(nb,indent=1)+'\n',encoding='utf-8')
with (root/'QUANTUM_BIGAN_PROGRESS.md').open('a',encoding='utf-8') as f:
    f.write(f"\n## Discriminator-update trial\n\nNotebook 05 completed: three D updates per G update after step 12, through step 64. Best step {trial['best_step']}; final validation MMD² {trial['history'][-1]['rbf_mmd_squared']:.4f}. See [SSW_STABILIZATION.md](SSW_STABILIZATION.md). No encoder added.\n")
print(text)
