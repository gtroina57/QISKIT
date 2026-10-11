# Quantum BiGAN development

## Stage 1 — validated

`GAN_01.ipynb` is unchanged. `GAN_01_latent4.ipynb` is its separate stage-one
variant. Only five original code cells were changed; saved outputs were cleared
to avoid displaying results from the original model.

- Four latent angles drawn independently from Uniform(-pi, pi) enter four Ry
  gates after the initial Hadamards.
- The original six-qubit ansatz, 84 trainable weights, 64 probability outputs,
  Gaussian target, discriminator, and adversarial losses are retained.
- SamplerQNN inputs and trainable weights are explicitly separated, with input
  gradients enabled through TorchConnector.
- Training averages four conditional distributions per step; evaluation uses
  32 latent draws. These are marginal probability distributions, not images.

Validation in the existing Windows DOUBT_QC environment:

```powershell
& 'C:/Users/giuse/.conda/envs/DOUBT_QC/python.exe' validate_latent_generator.py
```

`latent4_validation.json` records versions, the original notebook hash, and
results. The checks cover single and batched inference, normalization, latent
sensitivity, finite nonzero input and weight gradients, a weight update, two
training epochs, and final marginal evaluation. A fixed sampler seed is used
only by validation. The notebook retains the original sampler configuration.
All notebook code cells were also checked for Python syntax.

This is a smoke test, not evidence of training convergence or image quality.
The complete 50-epoch run and graphical rendering have not been validated.
`create_latent_notebook.py` reproduces the variant from the original notebook;
rerunning it overwrites the variant, so preserve any subsequent edits first.

## Recovered earlier dataset

The structured synthetic images are defined in `I_JEPA_05.ipynb`, cell 2.
`verify_dataset_source()` extracts that function without running JEPA training
and compares both images and factor labels exactly with the reused recipe.
The 256 training images (seed 42) and 128 test images (seed 123) match.
Images are `(1,8,8)` tensors with intensity `a*x+b*y`, where each slope is
uniform on `[0.2,0.8)`. Values can reach 1.6; images are not clipped to one.
No separate dataset files or an explicit "Synthetic Structured Worlds" title
were found in the project; this is the structured-image dataset recovered from
the earlier JEPA work.

## Stage 2 — validated quantum encoder

`quantum_bigan.py` adds a learned image-to-four-angle projection and a four-qubit
EstimatorQNN with four Z observables, 24 variational weights, and TorchConnector.
Outputs are four expectation values scaled by pi, matching the latent range.
`validate_quantum_encoder.py` checks the recovered dataset identity, shapes,
image sensitivity, finite nonzero image/preprocessing/quantum gradients, weight
updates, and preservation of the earlier notebooks. Results are saved in
`encoder_validation.json`.

## Stage 3 — validated BiGAN updates

The generator retains the six-qubit stage-one SamplerQNN and adds a learned
64-to-32-to-64 pixel decoder. It outputs continuous images in `[0,1.6]` from
the 64 probabilities; it does not sample pixels or images from bitstrings.
This makes the image generator hybrid quantum/classical, with differentiable
parameter-shift gradients through TorchConnector. The image decoder is an
explicit addition, not part of the original tutorial.

The discriminator extends the tutorial's 20-unit hidden layer to 68 inputs
(64 pixels and four latents), returning logits for stable BCE. Real pairs are
`(x,E(x))`; generated pairs are `(G(z),z)`. D uses detached G/E outputs. G/E
use a non-saturating loss with both labels flipped while D weights are frozen.
No reconstruction or factor-label supervision is used during training.

`validate_bigan_training.py` checks image shape/range, latent sensitivity and
gradients, pair logits, finite losses, and optimizer isolation. All generator,
decoder, encoder, and preprocessing parameters receive finite nonzero gradients
and update in the G/E step. Only D updates in its step. Results are saved in
`bigan_training_validation.json`.

## Stage 4 — short experiment and held-out evaluation

`GAN_02_BiGAN.ipynb` runs the stages sequentially with the current interpreter.
`run_quantum_bigan.py` defaults to two epochs on a fixed 32-image training subset
with batch size four and 1024 shots. Evaluation uses all 128 test images and
compares initial and trained models. Linear probes are fitted using the 256
training images and labels only. Metrics include factor MSE/R², image RBF MMD²,
reconstruction MSE, image variance, and planar-structure residual. Training-mean
and raw-pixel baselines are included. Test labels do not enter model training.

```powershell
& 'C:/Users/giuse/.conda/envs/DOUBT_QC/python.exe' validate_quantum_encoder.py
& 'C:/Users/giuse/.conda/envs/DOUBT_QC/python.exe' validate_bigan_training.py
& 'C:/Users/giuse/.conda/envs/DOUBT_QC/python.exe' run_quantum_bigan.py
```

`bigan_results/evaluation.json` records configuration, exact training indices,
metrics, versions, and dataset hashes. `quantum_bigan.pt` stores model weights;
`examples.pt` and `examples.png` preserve qualitative comparisons. The full
notebook can be executed with `execute_bigan_notebook.py`; notebook generation
scripts overwrite their respective variants, so save later edits before use.

The separate notebook completed execution with no error outputs. Its checkpoint
was loaded into a fresh model successfully, and the image comparison was
visually inspected. All original qGAN and JEPA notebook hashes were preserved.
The two-epoch, 32-image development run took approximately 176 seconds:

| Held-out metric | Initial model | After two epochs |
| --- | ---: | ---: |
| Encoder factor-probe MSE | 0.000205706 | 0.000036270 |
| Image RBF MMD² | 1.255370 | 1.232044 |
| Reconstruction MSE | 0.171194 | 0.152516 |
| Generated planar-structure residual MSE | 0.125289 | 0.126556 |

The mean-image reconstruction baseline is 0.019089, substantially better than
the trained model. Generated factor standard deviations remain about 0.025,
versus about 0.169 for real data. The generated images do not yet reproduce
the smooth planar worlds, despite improvement in the encoder probe. These
results validate the pipeline and expose remaining generation-quality work.

The short run validates execution and evaluation, not convergence. Longer
training on all 256 images is available using `--train-count 256 --epochs N`.
This dataset has only two factors despite the four-dimensional latent prior,
and a deterministic encoder cannot exactly match a full four-dimensional
uniform prior on a smooth two-dimensional data manifold. This limits what
adversarial convergence can establish. Raw pixels already encode the factors
linearly; this benchmark is not evidence of quantum advantage. Use repeat runs
across seeds before comparing model quality.

References: [Qiskit PyTorch qGAN tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/04_torch_qgan.html)
and [TorchConnector tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/05_torch_connector.html).
