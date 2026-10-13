# Original Qiskit qGAN reproduction

The original `GAN_01.ipynb` was executed in a separate notebook with every
original code cell unchanged, including `Sampler()` without a seed, 50 epochs,
the original optimizer/losses, and the original cumulative plots. Only an
introduction and a final diagnostic cell were added. Original notebook SHA-256
is unchanged: `34caa543d1d1b0c597fbc26302e7e6a09172befab4c2d934c85c5662f8326d63`.

Artifacts:

- `tutorial_reproduction/qiskit231_unseeded/GAN_01_reproduced.ipynb`: executed notebook.
- `tutorial_reproduction/qiskit231_unseeded/cell_15.png`: final training curves.
- `tutorial_reproduction/qiskit231_unseeded/cell_17.png`: original three CDF surfaces.
- `tutorial_reproduction/qiskit231_unseeded/results.json`: full metrics, versions, and probabilities.
- `tutorial_reproduction/published_training.png`: published training plot downloaded for comparison.

## What the published result actually shows

The [published training plot](https://qiskit-community.github.io/qiskit-machine-learning/_images/tutorials_04_torch_qgan_29_0.png)
ends at relative entropy approximately **0.42**, read visually from the plot.
It does not show relative entropy approaching zero. Its cumulative surfaces
illustrate an approximate distribution match, rather than exact equality of
the 64 point probabilities. See the [tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/04_torch_qgan.html).

## Reproduction results

| Quantity | Local unchanged run |
| --- | ---: |
| Last relative entropy plotted during training | 0.39448 |
| Relative entropy from a fresh final 1024-shot evaluation | 0.44800 |
| Final relative entropy from noiseless statevector probabilities | 0.37277 |
| Maximum absolute tutorial-style cumulative difference, sampled | 0.09687 |
| Maximum absolute tutorial-style cumulative difference, noiseless | 0.07540 |
| Noiseless total variation | 0.34688 |

The final training point uses sampled probabilities before the last optimizer
update. The final plot uses a new measurement sample after that update. They
need not have identical entropy. Noiseless diagnostics separately remove
measurement variation from the final learned state.

The training curve and cumulative surfaces are visually similar to the
published example. The earlier seeded baseline also had a maximum noiseless
cumulative difference of 0.07150 and KL 0.40013. Thus there is no demonstrated
large reproduction discrepancy requiring a software-version explanation.

The reproduction used Qiskit 2.3.1, Qiskit Machine Learning 0.9.1, PyTorch
2.14.1+cpu, Python 3.11.11, and Windows. The published version table lists
Qiskit 2.5.2, Machine Learning 0.9.1, Python 3.10.20, and Linux. This is a
faithful code reproduction in the local environment, not an identical-environment
or identical-random-draw reproduction. No package upgrades were made.

## Correction to the earlier investigation

The prior experiments remain valid measurements, but describing the local
baseline as failing to reproduce the tutorial was too strong. The tutorial's
own entropy curve has a similar nonzero endpoint. The inference that versions
or sampler seeding caused a failure was unconfirmed and is not supported by
this reproduction.

Two conclusions must be distinguished:

1. **Tutorial behavior reproduced:** approximate learning over 50 epochs, with
   visibly similar cumulative surfaces and relative entropy around 0.4.
2. **Stricter probability accuracy not achieved:** the individual point
   probabilities still differ substantially from the target.

Cumulative sums can cancel positive and negative pointwise errors. Consequently,
similar cumulative plots and nonzero KL/TV can coexist. Also, for fidelity the
tutorial's plots retain `np.cumsum` on the flattened 64-outcome vector, then
reshape to 8x8; this is not the usual joint two-dimensional CDF. The diagnostic
JSON reports a joint-CDF metric separately without altering the original plots.

The earlier requirement for near-zero Gaussian error was stricter than the
published example, not a reproduction requirement stated by the tutorial.
Further development should use explicit accuracy criteria and should not
assume the original tutorial is broken. Stricter fitting remains a separate
research question; the image BiGAN's quality also remains unestablished.

Re-run with the existing interpreter:

```powershell
& 'C:/Users/giuse/.conda/envs/DOUBT_QC/python.exe' reproduce_original_qgan.py --label another_unseeded_run
```

Unseeded sampling means repeated runs need not produce identical curves.
