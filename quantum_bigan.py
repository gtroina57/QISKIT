"""Incremental hybrid Quantum BiGAN; original qGAN notebooks remain untouched.

QNN integration follows the Qiskit TorchConnector tutorial:
https://qiskit-community.github.io/qiskit-machine-learning/tutorials/05_torch_connector.html
"""
from pathlib import Path
import ast
import hashlib
import json

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN

ROOT = Path(__file__).resolve().parent
LATENT_DIM = 4
IMAGE_SHAPE = (1, 8, 8)
IMAGE_MAX = 1.6


def make_images_and_labels(n, seed):
    """Exact tensor recipe from I_JEPA_05.ipynb, cell 2 (zero based)."""
    generator = torch.Generator().manual_seed(seed)
    a = torch.rand(n, 1, 1, generator=generator) * 0.6 + 0.2
    b = torch.rand(n, 1, 1, generator=generator) * 0.6 + 0.2
    x = torch.linspace(0, 1, 8).view(1, 1, 8)
    y = torch.linspace(0, 1, 8).view(1, 8, 1)
    images = (a * x + b * y).unsqueeze(1).float()
    labels = torch.cat([a.reshape(n, 1), b.reshape(n, 1)], dim=1)
    return images, labels


def verify_dataset_source():
    """Compare with the earlier notebook function without running JEPA training."""
    source = json.loads((ROOT / "I_JEPA_05.ipynb").read_text(encoding="utf-8"))
    cell = "".join(source["cells"][2]["source"])
    tree = ast.parse(cell)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "make_images_and_labels")
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(isolated, "I_JEPA_05-cell-2", "exec"), namespace)
    checks = {}
    for name, n, seed in (("train", 256, 42), ("test", 128, 123)):
        actual = make_images_and_labels(n, seed)
        expected = namespace["make_images_and_labels"](n, seed)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        checks[name] = {"count": n, "seed": seed,
                        "image_sha256": hashlib.sha256(actual[0].numpy().tobytes()).hexdigest(),
                        "label_sha256": hashlib.sha256(actual[1].numpy().tobytes()).hexdigest()}
    return {"source": "I_JEPA_05.ipynb, cell 2", "exact_match": True, "splits": checks}


def latent_prior(n, rng=None):
    return (2 * torch.rand(n, LATENT_DIM, generator=rng) - 1) * torch.pi


class QuantumEncoder(nn.Module):
    """Learned image-to-angle map plus four-qubit, four-observable QNN."""
    def __init__(self, seed=123456):
        super().__init__()
        self.image_to_angles = nn.Linear(64, 4)
        inputs = ParameterVector("x", 4)
        circuit = QuantumCircuit(4)
        for i, angle in enumerate(inputs):
            circuit.ry(angle, i)
        ansatz = efficient_su2(4, reps=2)
        circuit.compose(ansatz, inplace=True)
        # Qiskit Pauli strings put qubit zero at the right.
        observables = [SparsePauliOp("I" * (3 - i) + "Z" + "I" * i) for i in range(4)]
        qnn = EstimatorQNN(
            circuit=circuit, estimator=StatevectorEstimator(seed=seed),
            observables=observables, input_params=list(inputs),
            weight_params=list(ansatz.parameters), input_gradients=True,
            default_precision=0.0,
        )
        weights = np.random.default_rng(seed).uniform(-0.3, 0.3, qnn.num_weights)
        self.quantum = TorchConnector(qnn, initial_weights=weights)

    def forward(self, images):
        angles = torch.pi * torch.tanh(self.image_to_angles(images.flatten(1) / IMAGE_MAX))
        return torch.pi * self.quantum(angles)


class QuantumImageGenerator(nn.Module):
    """Stage-one probability QNN plus a learned continuous pixel decoder.

    The 64 probabilities are features, not sampled bitstrings or normalized
    images. Each z produces one image. Parameter-shift gradients flow through
    TorchConnector and the decoder; no discrete sampling estimator is needed.
    """
    def __init__(self, seed=123456, shots=1024):
        super().__init__()
        inputs = ParameterVector("z", LATENT_DIM)
        circuit = QuantumCircuit(6)
        circuit.h(circuit.qubits)
        for i, angle in enumerate(inputs):
            circuit.ry(angle, i)
        ansatz = efficient_su2(6, reps=6)
        circuit.compose(ansatz, inplace=True)
        qnn = SamplerQNN(
            circuit=circuit, sampler=StatevectorSampler(default_shots=shots, seed=seed),
            input_params=list(inputs), weight_params=list(ansatz.parameters),
            input_gradients=True, sparse=False,
        )
        weights = np.random.default_rng(seed).random(ansatz.num_parameters)
        self.quantum = TorchConnector(qnn, initial_weights=weights)
        self.decoder = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(0.2), nn.Linear(32, 64))

    def forward(self, z):
        probabilities = self.quantum(z)
        pixels = IMAGE_MAX * torch.sigmoid(self.decoder(64 * probabilities))
        return pixels.reshape(-1, *IMAGE_SHAPE)


class PairDiscriminator(nn.Module):
    """Tutorial's 20-unit hidden layer extended to 64 pixels + 4 latents."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(68, 20), nn.LeakyReLU(0.2), nn.Linear(20, 1))

    def forward(self, images, z):
        if images.shape[0] != z.shape[0] or z.shape[-1] != LATENT_DIM:
            raise ValueError("Expected equally sized image and four-dimensional latent batches")
        pair = torch.cat([images.flatten(1) / IMAGE_MAX, z / torch.pi], dim=1)
        return self.network(pair)


class QuantumBiGAN(nn.Module):
    def __init__(self, seed=123456, shots=1024):
        super().__init__()
        torch.manual_seed(seed)
        self.generator = QuantumImageGenerator(seed, shots)
        self.encoder = QuantumEncoder(seed)
        self.discriminator = PairDiscriminator()


def create_optimizers(model, lr=0.001):
    ge = torch.optim.Adam(list(model.generator.parameters()) + list(model.encoder.parameters()),
                          lr=lr, betas=(0.7, 0.999))
    d = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.7, 0.999))
    return ge, d


def discriminator_step(model, images, z, optimizer):
    """Real pair=(x,E(x)), generated pair=(G(z),z); detach G and E."""
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        inferred = model.encoder(images)
        generated = model.generator(z)
    real_logits = model.discriminator(images, inferred)
    fake_logits = model.discriminator(generated, z)
    bce = nn.functional.binary_cross_entropy_with_logits
    loss = 0.5 * (bce(real_logits, torch.ones_like(real_logits))
                  + bce(fake_logits, torch.zeros_like(fake_logits)))
    loss.backward()
    optimizer.step()
    return loss.item()


def generator_encoder_step(model, images, z, optimizer):
    """Non-saturating BiGAN update: flip both pair labels, freeze D weights."""
    optimizer.zero_grad(set_to_none=True)
    parameters = list(model.discriminator.parameters())
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        real_logits = model.discriminator(images, model.encoder(images))
        fake_logits = model.discriminator(model.generator(z), z)
        bce = nn.functional.binary_cross_entropy_with_logits
        loss = 0.5 * (bce(real_logits, torch.zeros_like(real_logits))
                      + bce(fake_logits, torch.ones_like(fake_logits)))
        loss.backward()
        optimizer.step()
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)
    return loss.item()


def train(model, train_images, epochs=2, batch_size=4, lr=0.001, seed=31415):
    model.train()
    ge_optimizer, d_optimizer = create_optimizers(model, lr)
    rng = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        indices = torch.randperm(len(train_images), generator=rng)
        d_total = ge_total = count = 0
        for start in range(0, len(indices), batch_size):
            batch = train_images[indices[start:start + batch_size]]
            z = latent_prior(len(batch), rng)
            d_loss = discriminator_step(model, batch, z, d_optimizer)
            ge_loss = generator_encoder_step(model, batch, z, ge_optimizer)
            if not np.isfinite([d_loss, ge_loss]).all():
                raise FloatingPointError("Nonfinite BiGAN loss")
            d_total += d_loss * len(batch)
            ge_total += ge_loss * len(batch)
            count += len(batch)
            print(f"Epoch {epoch + 1}/{epochs}, examples {count}/{len(train_images)}: "
                  f"D={d_loss:.5f}, GE={ge_loss:.5f}", flush=True)
        history.append({"epoch": epoch + 1, "discriminator_loss": d_total / count,
                        "generator_encoder_loss": ge_total / count})
    return history


@torch.no_grad()
def batched(module, values, batch_size=16):
    return torch.cat([module(values[i:i + batch_size]) for i in range(0, len(values), batch_size)])


def factor_probe(train_features, train_labels, test_features, test_labels):
    """Train-only least squares probe, consistent with earlier JEPA benchmark."""
    x = np.column_stack([train_features.numpy(), np.ones(len(train_features))])
    xt = np.column_stack([test_features.numpy(), np.ones(len(test_features))])
    coefficients = np.linalg.lstsq(x, train_labels.numpy(), rcond=None)[0]
    predicted = xt @ coefficients
    target = test_labels.numpy()
    residual = ((target - predicted) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    return {"test_mse": float(((target - predicted) ** 2).mean()),
            "test_r2_a_b": (1 - residual / total).tolist()}


def image_mmd(real, generated, bandwidth):
    """Biased RBF MMD squared; bandwidth fixed from training images only."""
    a, b = real.flatten(1).double(), generated.flatten(1).double()
    kernel = lambda x, y: torch.exp(-torch.cdist(x, y).square() / (2 * bandwidth ** 2))
    return (kernel(a, a).mean() + kernel(b, b).mean() - 2 * kernel(a, b).mean()).item()


@torch.no_grad()
def evaluate(model, train_images, train_labels, test_images, test_labels, seed=2718):
    model.eval()
    train_features = batched(model.encoder, train_images)
    test_features = batched(model.encoder, test_images)
    generated = batched(model.generator, latent_prior(len(test_images), torch.Generator().manual_seed(seed)))
    reconstructed = batched(model.generator, test_features)
    distances = torch.pdist(train_images.flatten(1).double())
    bandwidth = max(distances.median().item(), 1e-8)
    # Estimate image slope factors using the known grid, only as diagnostics.
    x = torch.linspace(0, 1, 8).view(1, 8).expand(8, 8).flatten()
    y = torch.linspace(0, 1, 8).view(8, 1).expand(8, 8).flatten()
    basis = torch.stack([x, y], dim=1)
    generated_factors = generated.flatten(1) @ torch.linalg.pinv(basis).T
    plane_projection = generated_factors @ basis.T
    report = {
        "encoder_factor_probe": factor_probe(train_features, train_labels, test_features, test_labels),
        "raw_pixel_factor_probe": factor_probe(train_images.flatten(1), train_labels,
                                               test_images.flatten(1), test_labels),
        "constant_factor_baseline_mse": (test_labels - train_labels.mean(0)).square().mean().item(),
        "reconstruction_mse": (reconstructed - test_images).square().mean().item(),
        "mean_image_baseline_mse": (train_images.mean(0) - test_images).square().mean().item(),
        "image_rbf_mmd_squared": image_mmd(test_images, generated, bandwidth),
        "mmd_bandwidth_from_train": bandwidth,
        "generated_pixel_variance": generated.var(0, unbiased=False).mean().item(),
        "real_pixel_variance": test_images.var(0, unbiased=False).mean().item(),
        "generated_plane_residual_mse": (generated.flatten(1) - plane_projection).square().mean().item(),
        "generated_factor_mean": generated_factors.mean(0).tolist(),
        "real_factor_mean": test_labels.mean(0).tolist(),
        "generated_factor_std": generated_factors.std(0, unbiased=False).tolist(),
        "real_factor_std": test_labels.std(0, unbiased=False).tolist(),
        "test_count": len(test_images),
    }
    return report, {"real": test_images[:8], "generated": generated[:8], "reconstructed": reconstructed[:8]}
