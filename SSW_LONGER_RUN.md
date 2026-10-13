# Longer SSW quantum-generator GAN run

`GAN_04_SSW_quantum_generator_longer.ipynb` continues the original twelve-step
experiment from `ssw_gan_results/final.pt` to **64 total generator updates**.
Models, both Adam states, minibatch RNG and fixed evaluation inputs were restored.
A zero-update resume test reproduced the original metrics and history before
the continuation. Unseeded quantum measurement draws cannot be replayed exactly.

The generator, decoder, discriminator, learning rates and batch size remain
unchanged. There is no encoder. SSW v1.0 observation noise remains at std 0.05;
no hardware error model, extra observation noise, clipping or data normalization
was introduced. The run processes 256 training-example draws in total, including
the original 48, from the 2,000-example training pool. This is not a full epoch.

## Result: longer unchanged training worsened the fit

| Validation diagnostic | Best: step 12 | Final: step 64 |
| --- | ---: | ---: |
| Fixed noiseless RBF MMD² | 0.19673 | 1.41338 |
| Feature mean MAE | 0.31134 | 1.33945 |
| Feature std MAE | 0.29732 | 0.26508 |
| Thresholded configurations covered | 5 | 2 |
| Clean-world logic residual MSE | 0.27443 | 2.29366 |
| Generated feature range | -0.587 to 1.574 | -0.460 to 4.694 |

Validation MMD reached about 1.50 around steps 44–56 and recovered only partly
by step 64. Thresholding the first four final features assigns 374 of 400
generated observations to configuration 1011 and the remaining 26 to 1111.
These are diagnostic assignments, not evidence that the continuous generated
vectors satisfy the corresponding world's equations.

Best checkpoint remains **step 12**. Its generator weights were verified equal
to the original best checkpoint. The final model was preserved separately;
it was not substituted for the selected best model.

A supplementary evaluation uses actual finite-shot generation at the same 400
latent inputs. Its validation MMD² is 0.19671 for the best checkpoint versus
1.40858 for the final checkpoint, supporting the deterministic comparison.
The final discriminator classifies about 63% of real validation samples as real
and all generated samples as fake in this draw. This gives balanced accuracy
about 0.816, but does not identify a unique cause of the generator's failure.

The worsening cannot be explained merely by noisy plotting: model evaluation
used fixed latent inputs throughout. Unlike the monitored Gaussian experiment,
this SSW run shows actual deterioration of the learned outputs. It does not
prove that every seed or longer schedule will fail, and the cause has not been
isolated by an ablation. More training alone was not a useful intervention here.

## Review and next experiment

Inspect `ssw_gan_longer_results/training.png`, `best_vs_final.png`, `report.json`
and `sampled_evaluation.json`. In the notebook's original two-panel sample
figure, generated samples are from the selected best checkpoint, not the final
one. The added best/final comparison distinguishes them explicitly, with the
same latent vectors and a shared unbounded feature-value scale. Real rows are
independent dataset samples.

Keep the step-12 checkpoint as the current reference, while recognizing its
sample diversity and logic are already inadequate. Before adding an encoder,
test one adversarial stabilization change at a time, such as additional
discriminator updates per generator update, retaining the same dataset and
generator. This intervention has not been tested in this run. Do not remove
the specified dataset noise or silently change SSW v1.0 to address instability.
