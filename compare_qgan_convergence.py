"""Matched qGAN convergence experiments without editing source notebooks."""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from qiskit.quantum_info import Statevector
from scipy.stats import entropy

ROOT = Path(__file__).resolve().parent


def build(latent, seed):
    name = "GAN_01_latent4.ipynb" if latent else "GAN_01.ipynb"
    notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    scope = {}
    for i in (0, 1, 2, 4, 7, 8, 9, 10, 11, 12):
        source = cells[i]
        if i == 0:
            source = source.replace("123456", str(seed))
        if i == 7:
            source = source.replace("Sampler()", f"Sampler(seed={seed})")
        exec(compile(source, f"{name}:cell{i}", "exec"), scope)
    return scope


def exact_probabilities(scope, z=None):
    weights = scope["generator"].weight.detach().numpy()
    binding = dict(zip(scope["ansatz"].parameters, weights))
    if z is not None:
        binding.update(zip(scope["latent_params"], z))
    return Statevector.from_instruction(scope["qc"].assign_parameters(binding)).probabilities()


def evaluate(scope, latent, fixed_z):
    if latent:
        # Averaging Ry(+pi/2) and Ry(-pi/2) gives the same channel as a
        # uniform angle on [-pi,pi]. Four independent rotations need 2^4
        # combinations. This exactly integrates the prior for this circuit.
        nodes = itertools.product((-np.pi / 2, np.pi / 2), repeat=4)
        p = np.mean([exact_probabilities(scope, z) for z in nodes], axis=0)
        fixed = np.mean([exact_probabilities(scope, z) for z in fixed_z], axis=0)
    else:
        p = fixed = exact_probabilities(scope)
    target = scope["prob_data"]
    assert np.isfinite(p).all() and np.isclose(p.sum(), 1)
    return {"kl_generated_to_target": float(entropy(p, target)),
            "total_variation": float(np.abs(p - target).sum() / 2),
            "fixed_latent_kl": float(entropy(fixed, target)),
            "fixed_vs_exact_tv": float(np.abs(fixed - p).sum() / 2)}, p


def run(variant, seed, epochs, out, batch_size, objective="adversarial"):
    latent = variant != "original"
    scope = build(latent, seed)
    g, d = scope["generator"], scope["discriminator"]
    go, do = scope["generator_optimizer"], scope["discriminator_optimizer"]
    rng = torch.Generator().manual_seed(seed + 1000)
    fixed_z = (2 * torch.rand(256, 4, generator=torch.Generator().manual_seed(98765)) - 1).numpy() * np.pi
    samples = torch.tensor(scope["grid_elements"], dtype=torch.float32)
    target = torch.tensor(scope["prob_data"], dtype=torch.float32).reshape(64, 1)
    ones, zeros = torch.ones(64, 1), torch.zeros(64, 1)
    initial_weights = g.weight.detach().numpy().tolist()
    initial_d = {n: p.detach().numpy().tolist() for n, p in d.named_parameters()}
    record = {"variant": variant, "seed": seed, "epochs": epochs,
              "objective": objective,
              "latent_batch_size": batch_size if latent else None,
              "initial_weights": initial_weights, "initial_discriminator": initial_d,
              "evaluation": [], "losses": []}
    start = time.perf_counter()
    metrics, p = evaluate(scope, latent, fixed_z)
    record["evaluation"].append({"epoch": 0, **metrics})
    record["initial_probabilities"] = p.tolist()
    for epoch in range(1, epochs + 1):
        disc = d(samples)
        z = (2 * torch.rand(batch_size, 4, generator=rng) - 1) * torch.pi if latent else torch.tensor([])
        dist = g(z)
        dist = (dist.mean(0) if latent else dist).reshape(64, 1)
        go.zero_grad()
        if objective == "direct-kl":
            gl = torch.sum(dist * (torch.log(dist.clamp_min(1e-8)) - torch.log(target)))
        else:
            gl = scope["adversarial_loss"](disc, ones, dist)
        gl.backward(retain_graph=objective == "adversarial")
        go.step()
        if objective == "adversarial":
            do.zero_grad()
            dl = (scope["adversarial_loss"](disc, ones, target)
                  + scope["adversarial_loss"](disc, zeros, dist.detach())) / 2
            dl.backward()
            do.step()
        else:
            dl = torch.tensor(0.0)  # Discriminator not trained in this diagnostic.
        assert torch.isfinite(gl) and torch.isfinite(dl)
        record["losses"].append({"epoch": epoch, "generator": gl.item(), "discriminator": dl.item(),
                                 "training_batch_kl": float(entropy(dist.detach().numpy().ravel(), scope["prob_data"]))})
        if epoch % 5 == 0 or epoch == epochs:
            metrics, p = evaluate(scope, latent, fixed_z)
            record["evaluation"].append({"epoch": epoch, **metrics})
            print(f"{variant} [{objective}] seed={seed} epoch={epoch}/{epochs}: "
                  f"exact KL={metrics['kl_generated_to_target']:.5f}, TV={metrics['total_variation']:.5f}", flush=True)
        record["elapsed_seconds"] = time.perf_counter() - start
        (out / f"{variant}_{seed}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record["final_probabilities"] = p.tolist()
    record["target_probabilities"] = scope["prob_data"].tolist()
    (out / f"{variant}_{seed}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    torch.save({"generator": g.state_dict(), "discriminator": d.state_dict()}, out / f"{variant}_{seed}.pt")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123456])
    parser.add_argument("--variants", nargs="+", choices=["original", "latent4", "latent16"], default=["original", "latent4"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "convergence_results")
    parser.add_argument("--objective", choices=["adversarial", "direct-kl"], default="adversarial")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if args.objective == "direct-kl" and args.output_dir.resolve() == (ROOT / "convergence_results").resolve():
        parser.error("Use a separate --output-dir for the direct-KL diagnostic")
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ("GAN_01.ipynb", "GAN_01_latent4.ipynb")
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in names}
    results = []
    for seed in args.seeds:
        for variant in args.variants:
            results.append(run(variant, seed, args.epochs, args.output_dir, 16 if variant == "latent16" else 4, args.objective))
    for seed in args.seeds:
        paired = [r for r in results if r["seed"] == seed]
        assert all(r["initial_weights"] == paired[0]["initial_weights"] for r in paired)
        assert all(r["initial_discriminator"] == paired[0]["initial_discriminator"] for r in paired)
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    print("Matched initial parameters verified; source notebooks unchanged.", flush=True)


if __name__ == "__main__":
    main()
