"""Show fixed-input best/final samples from the completed SSW continuation."""
from pathlib import Path
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ssw_quantum_gan import QuantumGenerator

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--results', default='ssw_gan_longer_results')
out=root/parser.parse_args().results
torch.set_num_threads(2)
report=json.loads((out/'report.json').read_text())
real=torch.tensor(np.load(root/'ssw_v1.npz')['val_iid_x'][:24])
shown=[real]
labels=['Real SSW observations']
states=[]
for name in ['best','final']:
    state=torch.load(out/f'{name}.pt',weights_only=True)
    states.append(state)
    model=QuantumGenerator(report['seed'])
    model.load_state_dict(state['generator'])
    shown.append(model.exact(state['evaluation_z'][:24]))
    labels.append(f'{name.capitalize()} checkpoint: step {state["step"]}')
vmin=min(real.min().item(),*(s['validation_metrics']['generated_min'] for s in states))
vmax=max(real.max().item(),*(s['validation_metrics']['generated_max'] for s in states))
fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
for ax,x,title in zip(axes,shown,labels):
    image=ax.imshow(x.numpy(),aspect='auto',vmin=vmin,vmax=vmax,cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('Feature')
    ax.set_xticks(range(8),[f'x{i+1}' for i in range(8)])
    ax.set_ylabel('Sample')
fig.colorbar(image,ax=list(axes),label='Unclipped feature value',shrink=0.8)
fig.suptitle('Fixed noiseless generation: same latent inputs for best/final; real rows are independent')
fig.savefig(out/'best_vs_final.png',dpi=150,bbox_inches='tight')
plt.close(fig)
print('Saved best-versus-final comparison.')
