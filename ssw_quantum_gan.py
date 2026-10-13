"""Quantum-generator SSW GAN. No encoder or BiGAN objective."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from ssw_dataset import clean_features


def prior(n, rng):
    return (2 * torch.rand(n, 4, generator=rng) - 1) * torch.pi


class QuantumGenerator(nn.Module):
    def __init__(self, seed=123456):
        super().__init__()
        self.inputs = ParameterVector("z", 4)
        self.circuit = QuantumCircuit(6)
        self.circuit.h(self.circuit.qubits)
        for i, angle in enumerate(self.inputs):
            self.circuit.ry(angle, i)
        ansatz = efficient_su2(6, reps=6)
        self.weights = list(ansatz.parameters)
        self.circuit.compose(ansatz, inplace=True)
        qnn = SamplerQNN(circuit=self.circuit, sampler=StatevectorSampler(),
                         input_params=list(self.inputs), weight_params=self.weights,
                         sparse=False, input_gradients=True)
        self.quantum = TorchConnector(qnn, np.random.default_rng(seed).random(84))
        # Linear output: SSW noise is not clipped to [0,1].
        self.decoder = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(0.2), nn.Linear(32, 8))

    def forward(self, z):
        return self.decoder(64 * self.quantum(z))

    @torch.no_grad()
    def exact(self, z):
        weights = dict(zip(self.weights, self.quantum.weight.detach().numpy()))
        probabilities = [Statevector.from_instruction(self.circuit.assign_parameters(
            {**weights, **dict(zip(self.inputs, row.tolist()))})).probabilities() for row in z]
        return self.decoder(64 * torch.tensor(np.array(probabilities), dtype=torch.float32))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(8, 20), nn.LeakyReLU(0.2), nn.Linear(20, 1))

    def forward(self, x):
        return self.layers(x)


def step(g, d, go, do, real, z, discriminator_updates=1):
    bce = nn.functional.binary_cross_entropy_with_logits
    assert discriminator_updates >= 1
    with torch.no_grad():
        fake = g(z)
    for _ in range(discriminator_updates):
        do.zero_grad(set_to_none=True)
        a, b = d(real), d(fake)
        dl = (bce(a, torch.ones_like(a)) + bce(b, torch.zeros_like(b))) / 2
        dl.backward()
        do.step()
    go.zero_grad(set_to_none=True)
    do.zero_grad(set_to_none=True)
    try:
        for p in d.parameters():
            p.requires_grad_(False)
        logits = d(g(z))
        gl = bce(logits, torch.ones_like(logits))
        gl.backward()
        gradient_norms = {}
        for name, module in (("quantum", g.quantum), ("decoder", g.decoder)):
            gradients = [p.grad.detach() for p in module.parameters() if p.grad is not None]
            gradient_norms[f"{name}_gradient_norm"] = sum(
                v.double().square().sum().item() for v in gradients) ** 0.5
            assert gradients and all(torch.isfinite(v).all() for v in gradients)
        go.step()
    finally:
        for p in d.parameters():
            p.requires_grad_(True)
    assert torch.isfinite(dl) and torch.isfinite(gl)
    return {"discriminator_loss": dl.item(), "generator_loss": gl.item(), **gradient_norms,
            "discriminator_accuracy": ((a.detach() >= 0).float().mean() + (b.detach() < 0).float().mean()).item()/2}


def metrics(real, fake, bandwidth):
    a, b = real.double(), fake.double()
    kernel = lambda x,y: torch.exp(-torch.cdist(x,y).square()/(2*bandwidth**2))
    mmd = kernel(a,a).mean()+kernel(b,b).mean()-2*kernel(a,b).mean()
    bits = (fake[:, :4] >= 0.5).numpy().astype(np.int8)
    clean = torch.tensor(clean_features(bits))
    codes = bits @ np.array([8,4,2,1])
    counts = np.bincount(codes, minlength=16)
    return {"rbf_mmd_squared": mmd.item(),
            "feature_mean_mae": (real.mean(0)-fake.mean(0)).abs().mean().item(),
            "feature_std_mae": (real.std(0, unbiased=False)-fake.std(0, unbiased=False)).abs().mean().item(),
            "thresholded_configuration_counts": counts.tolist(),
            "covered_configurations": int((counts>0).sum()),
            "logic_residual_mse": (fake-clean).square().mean().item(),
            "generated_min": fake.min().item(), "generated_max": fake.max().item()}


def train_run(arrays, out, steps=12, batch_size=4, seed=123456, resume=None, discriminator_updates=1,
              generator_lr=None):
    if generator_lr is not None and (not np.isfinite(generator_lr) or generator_lr <= 0):
        raise ValueError("generator_lr must be finite and positive")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    g, d = QuantumGenerator(seed), Discriminator()
    go = torch.optim.Adam(g.parameters(), lr=0.001, betas=(0.7,0.999))
    do = torch.optim.Adam(d.parameters(), lr=0.001, betas=(0.7,0.999))
    real = torch.tensor(arrays["train_x"])
    val = torch.tensor(arrays["val_iid_x"])
    eval_z = prior(400, torch.Generator().manual_seed(98765))
    rng = torch.Generator().manual_seed(seed+1)
    # Bandwidth derived from training data only, fixed for every split.
    bandwidth = torch.pdist(real[:512].double()).median().item()
    history, losses = [], []
    best = float("inf")
    start_step = 0
    if resume is not None:
        resume = Path(resume)
        assert resume.parent.resolve() != out.resolve(), "Use a separate directory for resumed runs"
        state = torch.load(resume, weights_only=True)
        previous = json.loads((resume.parent / "report.json").read_text(encoding="utf-8"))
        assert previous["seed"] == seed and previous["batch_size"] == batch_size
        g.load_state_dict(state["generator"])
        d.load_state_dict(state["discriminator"])
        go.load_state_dict(state["generator_optimizer"])
        do.load_state_dict(state["discriminator_optimizer"])
        eval_z = state["evaluation_z"]
        rng.set_state(state["minibatch_rng"])
        torch.set_rng_state(state["torch_rng"])
        start_step = state["step"]
        assert steps >= start_step
        history = [r for r in previous["history"] if r["step"] <= start_step]
        losses = [r for r in previous["losses"] if r["step"] <= start_step]
        prior_best = torch.load(resume.parent / "best.pt", weights_only=True)
        assert prior_best["step"] <= start_step
        best = prior_best["validation_metrics"]["rbf_mmd_squared"]
        torch.save(prior_best, out / "best.pt")
        restored_score = metrics(val, g.exact(eval_z), bandwidth)
        assert abs(restored_score["rbf_mmd_squared"] - state["validation_metrics"]["rbf_mmd_squared"]) < 1e-10
        print(f"Restored models, optimizers and RNGs at step {start_step}; validation score verified", flush=True)
    # Apply after restoring Adam: loading optimizer state restores its old rate.
    if generator_lr is not None:
        for group in go.param_groups:
            group["lr"] = generator_lr
    def evaluate_and_save(index):
        nonlocal best
        fake = g.exact(eval_z)
        score = metrics(val, fake, bandwidth)
        if not history or history[-1]["step"] != index:
            history.append({"step": index, **score})
        state = {"step": index, "generator": copy.deepcopy(g.state_dict()),
                 "discriminator": copy.deepcopy(d.state_dict()), "generator_optimizer": go.state_dict(),
                 "discriminator_optimizer": do.state_dict(), "evaluation_z": eval_z,
                 "validation_metrics": score, "torch_rng": torch.get_rng_state(),
                 "minibatch_rng": rng.get_state(), "sampler_seed": None}
        torch.save(state, out/"latest.pt")
        if score["rbf_mmd_squared"] < best:
            best = score["rbf_mmd_squared"]
            torch.save(state, out/"best.pt")
        (out/"history.json").write_text(json.dumps(history,indent=2)+"\n",encoding="utf-8")
        print(f"Step {index}/{steps}: validation MMD²={score['rbf_mmd_squared']:.5f}",flush=True)
        return fake
    evaluate_and_save(start_step)
    for index in range(start_step+1, steps+1):
        indices = torch.randint(len(real), (batch_size,), generator=rng)
        losses.append({"step": index, **step(g,d,go,do,real[indices],prior(batch_size,rng),discriminator_updates)})
        print(f"Training step {index}/{steps}: {losses[-1]}",flush=True)
        if index % 4 == 0 or index == steps:
            evaluate_and_save(index)
    torch.save(torch.load(out/"latest.pt",weights_only=True), out/"final.pt")
    final_state = torch.load(out/"final.pt", weights_only=True)
    best_state = torch.load(out/"best.pt", weights_only=True)
    g.load_state_dict(best_state["generator"])
    selected_fake = g.exact(eval_z)
    checked = metrics(val,selected_fake,bandwidth)
    assert abs(checked["rbf_mmd_squared"]-best_state["validation_metrics"]["rbf_mmd_squared"])<1e-10
    tests = {split: metrics(torch.tensor(arrays[f"{split}_x"]),selected_fake,bandwidth)
             for split in ("test_iid","test_ood")}
    report = {"dataset":"SSW v1.0", "encoder":False, "steps":steps, "batch_size":batch_size,
              "training_pool_size":len(real), "processed_training_examples":steps*batch_size,
              "resumed_from":str(resume) if resume is not None else None,
              "new_steps":steps-start_step,
              "generator_learning_rate":go.param_groups[0]["lr"],
              "discriminator_learning_rate":do.param_groups[0]["lr"],
              "generator_learning_rate_override":generator_lr,
              "discriminator_updates_per_generator":discriminator_updates,
              "discriminator_updates_changed_after_step":start_step if resume is not None else 0,
              "full_training_epoch":False, "seed":seed, "best_step":best_state["step"],
              "selection":"val_iid RBF MMD²", "bandwidth_from_train":bandwidth,
              "history":history, "losses":losses, "selected_test_metrics":tests,
              "note":"Short integration run; no convergence or OOD generation claim. Unconditional generator is trained on IID configurations only."}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    torch.save({"selected_generated":selected_fake,"validation_real":val},out/"examples.pt")
    g.load_state_dict(final_state["generator"])
    return report
