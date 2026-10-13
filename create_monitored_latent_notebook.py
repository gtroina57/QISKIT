"""Preserve the existing latent notebook and add diagnostics to a separate copy."""
import copy
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parent
n = nbformat.read(root / "GAN_01_latent4.ipynb", as_version=4)
cells = [c for c in n.cells if c.cell_type == "code"]
setup = '''from latent_training_monitor import LatentTrainingMonitor
from pathlib import Path

# Own RNG for evaluation; random training inputs and sampler remain unchanged.
monitor = LatentTrainingMonitor(globals(), Path("latent_monitor_results"), evaluation_count=64)
monitor.record(0)
'''
training = cells[14]
assert "    plot_training_progress()" in training.source
training.source = training.source.replace("    plot_training_progress()", "    monitor.record(epoch + 1, training_batch_kl=entropy_value)\n    plot_training_progress()")
training.source += '\nmonitor_summary = monitor.finish()\nprint(monitor_summary)\n'
# Insert after optimizers and before progress plotting/training.
n.cells.insert(n.cells.index(cells[12]) + 1, nbformat.v4.new_code_cell(setup))
n.cells.insert(0, nbformat.v4.new_markdown_cell('''# Latent qGAN with fixed evaluation and checkpoints

Separate copy of `GAN_01_latent4.ipynb`. Circuit, latent training batch (4),
random prior, unseeded sampler, optimizer settings, losses and 50-epoch update
order are unchanged. Added diagnostics do not consume the training RNG or
quantum measurement samples.

After each epoch, evaluate the same 64 latent vectors using noiseless circuit
probabilities. Also evaluate the exact uniform-prior marginal using the validated
16-point quadrature for this circuit. Save best, latest and final checkpoints
under `latent_monitor_results`, including generator/discriminator weights,
optimizer states, fixed evaluation inputs and Torch RNG state. Best is selected
by exact marginal KL, not the noisy training-batch curve. Epoch 0 is eligible.

Training-batch KL is measured before the update; fixed and exact evaluation are
measured after it. The final monitoring figure makes these distinctions explicit.
Checkpoint RNG state does not reproduce the unseeded sampler's random draws.
No early stopping or automatic loading of the best model is applied. Final
tutorial plots still show the final model, not the best checkpoint.
'''))
for c in n.cells:
    if c.cell_type == "code":
        c.execution_count = None
        c.outputs = []
nbformat.write(n, root / "GAN_01_latent4_monitored.ipynb")
print("Created GAN_01_latent4_monitored.ipynb")
