"""Supplement deterministic SSW diagnostics with actual finite-shot generation."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from ssw_quantum_gan import QuantumGenerator, Discriminator, metrics

parser=argparse.ArgumentParser()
parser.add_argument('--results',type=Path,default=Path('ssw_gan_longer_results'))
args=parser.parse_args()
root=Path(__file__).resolve().parent
torch.set_num_threads(2)
arrays=dict(np.load(root/'ssw_v1.npz',allow_pickle=False))
report=json.loads((args.results/'report.json').read_text())
states={name:torch.load(args.results/f'{name}.pt',weights_only=True) for name in ['best','final']}
summary={'seed_note':'Training and this finite-shot diagnostic use an unseeded sampler.',
         'versions':{p:version(p) for p in ['qiskit','qiskit-machine-learning','torch','numpy']},
         'checkpoints':{}}
for name,state in states.items():
    model=QuantumGenerator(report['seed'])
    model.load_state_dict(state['generator'])
    discriminator=Discriminator()
    discriminator.load_state_dict(state['discriminator'])
    with torch.no_grad():
        sampled=model(state['evaluation_z'])
        real_logits=discriminator(torch.tensor(arrays['val_iid_x']))
        fake_logits=discriminator(sampled)
    summary['checkpoints'][name]={'step':state['step'],
        'noiseless_validation':state['validation_metrics'],
        'sampled_validation':metrics(torch.tensor(arrays['val_iid_x']),sampled,report['bandwidth_from_train'])}
    summary['checkpoints'][name]['discriminator']={
        'real_classified_real_rate':(real_logits>=0).float().mean().item(),
        'fake_classified_real_rate':(fake_logits>=0).float().mean().item(),
        'balanced_accuracy':((real_logits>=0).float().mean()+(fake_logits<0).float().mean()).item()/2}
(args.results/'sampled_evaluation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
