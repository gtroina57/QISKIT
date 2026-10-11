"""Stage two: dataset identity and quantum encoder gradient checks."""
import hashlib
import json
from pathlib import Path
import torch
from quantum_bigan import QuantumEncoder, make_images_and_labels, verify_dataset_source

root = Path(__file__).resolve().parent
torch.manual_seed(123456)
hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
          for p in ("GAN_01.ipynb", "GAN_01_latent4.ipynb", "I_JEPA_05.ipynb")}
dataset = verify_dataset_source()
print("Earlier JEPA dataset exactly reproduced", flush=True)
encoder = QuantumEncoder()
images, _ = make_images_and_labels(2, 42)
images.requires_grad_(True)
encoded = encoder(images)
assert encoded.shape == (2, 4) and torch.isfinite(encoded).all()
assert encoded.abs().max() <= torch.pi + 1e-6
assert not torch.allclose(encoded[0], encoded[1])
qnn = encoder.quantum.neural_network
assert qnn.circuit.num_qubits == 4 and qnn.num_inputs == 4
assert qnn.num_weights == 24 and qnn.output_shape == (4,)
assert set(qnn.input_params).isdisjoint(qnn.weight_params)
optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
before = {n: p.detach().clone() for n, p in encoder.named_parameters()}
encoded.square().mean().backward()
assert images.grad is not None and torch.isfinite(images.grad).all() and images.grad.abs().sum() > 0
for n, p in encoder.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n
optimizer.step()
for n, p in encoder.named_parameters():
    assert not torch.equal(before[n], p), n
for p, digest in hashes.items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == digest
report = {"stage": 2, "status": "passed", "dataset": dataset,
          "encoder_qubits": 4, "quantum_weights": 24, "latent_shape": list(encoded.shape),
          "checks": ["exact prior dataset reproduction", "image sensitivity", "finite outputs",
                     "image, preprocessing and quantum gradients", "optimizer updates", "notebooks unchanged"]}
(root / "encoder_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
