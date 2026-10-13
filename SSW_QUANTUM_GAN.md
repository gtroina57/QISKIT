# SSW quantum-generator GAN, without encoder

Open `GAN_03_SSW_quantum_generator.ipynb`. It has been executed end to end.
All prior Gaussian and image-BiGAN notebooks are preserved.

The dataset is SSW v1.0 from the supplied specification, copied into
`SSW_SPECIFICATION_v1.txt`. The generator code, `ssw_v1.npz`, and metadata are
alongside it. The original companion implementation was not found in the
workspace; this reconstruction records its exact configuration ordering and
RNG operations and does not claim byte equality to an unavailable reference.
Formula, shape, split, OOD separation, noise and repeatability checks passed.

## Model

Four Uniform(-pi,pi) angles enter the retained six-qubit circuit, with four Ry
input gates, six efficient_su2 repetitions and 84 quantum weights. SamplerQNN
and TorchConnector produce 64 probability features. A learned 64→32→8 decoder
produces one observation per latent vector. Its output is linear, allowing the
unclipped noisy feature range. The decoder receives probabilities multiplied
by 64; SSW observations are not normalized or scaled.

The discriminator is 8→20→1 with logits. It receives observations only.
There is no encoder or image representation. Generator/decoder parameters:
2428; discriminator parameters: 201. Adam uses lr 0.001 and betas (0.7,0.999).
The quantum sampler has 1024 shots and no fixed sampler seed.

## Completed development run

Twelve optimizer steps with batch size four draw from the 2,000-sample training
pool with replacement. Only 48 training examples were processed: this is not
a full training epoch. Validation uses 400 fixed latent vectors and noiseless
circuit probabilities, followed by the learned decoder. This deterministic
diagnostic is distinct from finite-shot generator sampling during training.
Bandwidth is selected from training observations only. Checkpoint selection
uses val_iid MMD; neither test split selects parameters.

| Validation diagnostic | Initial | Step 12 |
| --- | ---: | ---: |
| RBF MMD squared | 0.48504 | 0.19673 |
| Feature-mean MAE | 0.54919 | 0.31134 |
| Feature-standard-deviation MAE | 0.29909 | 0.29732 |
| Thresholded configurations covered | 1 | 5 |
| Clean-world logic residual MSE | 0.16540 | 0.27443 |

Best checkpoint is step 12 and was reloaded to verify its validation score.
MMD and feature means improve, but feature variance barely improves and logic
consistency worsens. Generated observations do not yet reproduce the dataset's
binary modes and feature relationships. This is a functioning integration
baseline, not evidence of convergence or useful OOD generation. The unconditional
model is trained only on the twelve IID configurations; the held-out test
comparison is exploratory, not a target distribution learned during training.

`ssw_gan_validation.json` records input/quantum/decoder gradient checks, latent
sensitivity, generator updates and deterministic evaluation. Notebook execution
stops if validation fails. All cells completed without error.

## Artifacts

`ssw_gan_results` stores `best.pt`, `latest.pt`, `final.pt`, `history.json`,
`report.json`, `examples.pt`, `training.png` and `samples.png`. Checkpoints
include generator/discriminator and optimizer states. Unseeded measurement draws
cannot be replayed exactly. Heatmaps display eight-feature vectors, not images,
with a shared intensity scale; their real/generated rows are independent samples.

The next experiment can increase STEPS in the notebook while preserving model
settings. Establish diversity and logical structure alongside MMD before
changing architecture or adding an encoder. No further stage was run here.
