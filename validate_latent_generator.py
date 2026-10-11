"""Validate the actual stage-one notebook with the installed Qiskit stack."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLBACKEND", "Agg")
root = Path(__file__).resolve().parent
original_hash = hashlib.sha256((root / "GAN_01.ipynb").read_bytes()).hexdigest()
notebook = json.loads((root / "GAN_01_latent4.ipynb").read_text(encoding="utf-8"))
cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
scope = {}
start = time.perf_counter()
for index in (0, 1, 2, 4, 7, 8, 9, 10, 11, 12):
    source = cells[index]
    if index == 7:
        source = source.replace("Sampler()", "Sampler(seed=123456)")
    exec(compile(source, f"notebook-cell-{index}", "exec"), scope)

torch = scope["torch"]
generator = scope["generator"]
qnn = generator.neural_network
assert qnn.num_inputs == 4
assert qnn.num_weights == 84
assert set(qnn.input_params).isdisjoint(qnn.weight_params)
assert set(qnn.input_params) | set(qnn.weight_params) == set(scope["qc"].parameters)

# Seed the finite-shot sampler so differences are due to the latent input.
z = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.2, -0.7, 0.9, -1.1]], requires_grad=True)
probabilities = generator(z)
assert probabilities.shape == (2, 64)
assert torch.isfinite(probabilities).all() and (probabilities >= 0).all()
assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-6)
assert not torch.allclose(probabilities[0], probabilities[1], atol=1e-3)
assert generator(z[0].detach()).shape == (64,)
before = generator.weight.detach().clone()
loss = (probabilities * torch.linspace(-1, 1, 64)).sum()
loss.backward()
assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
assert generator.weight.grad is not None
assert torch.isfinite(generator.weight.grad).all() and generator.weight.grad.abs().sum() > 0
scope["generator_optimizer"].step()
assert not torch.equal(before, generator.weight.detach())
latent_difference = (probabilities[0] - probabilities[1]).abs().sum().item()

# Fresh models and optimizer state for an actual short tutorial training run.
for index in (10, 12):
    exec(compile(cells[index], f"notebook-cell-{index}", "exec"), scope)
scope["plot_training_progress"] = lambda: None
training = cells[14].replace("n_epochs = 50", "n_epochs = 2")
exec(compile(training, "notebook-training", "exec"), scope)
for name in ("generator_loss_values", "discriminator_loss_values", "entropy_values"):
    assert len(scope[name]) == 2
    assert all(scope["np"].isfinite(scope[name]))
exec(compile(cells[15], "notebook-evaluation", "exec"), scope)
assert scope["generated_probabilities"].shape == (64,)
assert abs(scope["generated_probabilities"].sum() - 1) < 1e-6
assert hashlib.sha256((root / "GAN_01.ipynb").read_bytes()).hexdigest() == original_hash
from importlib.metadata import version
report = {
    "stage": 1, "status": "passed", "original_sha256": original_hash,
    "versions": {p: version(p) for p in ("qiskit", "qiskit-machine-learning", "torch", "numpy", "scipy")},
    "latent_dim": qnn.num_inputs, "trainable_weights": qnn.num_weights,
    "output_shape": list(probabilities.shape), "latent_distribution_l1_difference": latent_difference,
    "checks": ["parameter separation", "single and batch forward", "normalized probabilities", "latent sensitivity", "input and weight gradients", "optimizer update", "two training epochs", "latent-marginal evaluation", "original unchanged"],
    "generator_loss": scope["generator_loss_values"],
    "discriminator_loss": scope["discriminator_loss_values"],
    "relative_entropy": scope["entropy_values"],
    "elapsed_seconds": time.perf_counter() - start,
}
(root / "latent4_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
