# Fixed evaluation and checkpoints for the latent qGAN

Open `GAN_01_latent4_monitored.ipynb` and run it from the project directory.
This is a separate copy: `GAN_01.ipynb` and `GAN_01_latent4.ipynb` are preserved.

The circuit, four-dimensional prior, random training batch of four vectors,
unseeded 1024-shot sampler, optimizer settings, losses and 50-epoch update
order are retained. The changes add monitoring only; there is no early stopping
and no automatic replacement of the final weights by the best weights.

## Reading the curves

- **Training-batch KL:** the original curve, from four new random latent vectors
  and shot sampling, measured before that epoch's optimizer update.
- **Fixed-input KL:** the same 64 independent latent vectors evaluated after
  every epoch using noiseless statevector probabilities. The evaluation vectors
  have their own RNG and do not consume the training RNG or sampler draws.
- **Exact-prior KL/TV:** noiseless probabilities averaged over the uniform prior
  using the independently validated sixteen-point rule for this circuit. This
  avoids finite-latent evaluation error as well as measurement error.

The fixed set is a deterministic diagnostic, not a held-out image dataset.
The exact prior evaluation depends on the circuit encoding each angle once,
before the ansatz; changes to that encoding require validating the rule again.

If the noisy curve spikes while fixed/exact curves remain smooth, sampling
variation explains those spikes. Sustained worsening of fixed/exact scores
would instead support deterioration of the learned distribution. Small late
fluctuations are not by themselves stable convergence followed by divergence.
No single-seed run establishes reliable behavior across seeds.

## Artifacts

`latent_monitor_results` contains:

- `history.json`: epoch-zero and every post-update evaluation, plus training-batch KL.
- `best.pt`: lowest exact-prior KL, including epoch zero if it is best.
- `latest.pt`: latest completed epoch, useful if a run stops early.
- `final.pt`: final completed model, saved after the training loop.
- `summary.json`: best/final epochs and scores, including any final deterioration.
- `monitoring.png`: noisy versus fixed/exact curves.

Checkpoints contain generator and discriminator weights, both Adam states,
the fixed evaluation vectors, Torch RNG state and metrics. They allow restoring
model/optimizer state, but the unseeded sampler's measurement draws cannot be
replayed exactly.

The original tutorial plots at the end still show the final generator, using
its original 32-vector evaluation. The monitoring plot uses deterministic
evaluation and explicitly distinguishes best and final epochs.

## Validation and execution

The actual monitored setup/training cells passed a two-epoch smoke test. The
best checkpoint reloaded and reproduced its exact marginal KL. The monitor
asserts that evaluation leaves generator weights and the default Torch RNG
unchanged. Creating its fixed vectors also leaves that RNG unchanged.

```powershell
& 'C:/Users/giuse/.conda/envs/DOUBT_QC/python.exe' run_monitored_latent.py --epochs 2 --smoke
& 'C:/Users/giuse/.conda/envs/DOUBT_QC/python.exe' execute_monitored_latent.py
```

The smoke test uses `latent_monitor_smoke`, keeping the full-run artifacts
separate. Re-running the notebook or executor overwrites that run's monitoring
files. Copy a run's result directory first if it must be preserved.

## Completed 50-epoch run

All notebook code cells executed without errors, and both source notebooks
remained unchanged. The best checkpoint and both optimizer states reloaded
successfully; its exact score matched the saved score.

| Deterministic metric | Initial | Final (epoch 50) |
| --- | ---: | ---: |
| Exact-prior KL | 0.51008 | 0.30105 |
| Fixed 64-vector KL | 0.54688 | 0.32353 |
| Exact-prior total variation | 0.38374 | 0.29502 |

Best epoch is **50**, so the best and final models coincide in this run. The
largest exact KL increase above the previous best during training was only
0.00090. By comparison, the standard deviation of training-batch KL minus the
corresponding pre-update exact KL was about 0.092. These residuals combine
finite latent-batch and measurement effects; this experiment does not separate
their individual contributions.

The plot supports overall improvement of the learned marginal distribution,
despite large spikes in the original training-batch curve. It does not show
sustained learned-distribution divergence in this run. This is one new stochastic
run, not reconstruction of the exact earlier screenshot, nor proof of stable
behavior across every seed. The final distribution still approximates rather
than exactly matches the Gaussian.
