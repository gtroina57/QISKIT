"""Stage three: pair shapes and adversarial gradient/update isolation."""
import json
from pathlib import Path
import torch
from quantum_bigan import (
    QuantumBiGAN, create_optimizers, discriminator_step,
    generator_encoder_step, latent_prior, make_images_and_labels,
)

root = Path(__file__).resolve().parent
model = QuantumBiGAN()
images, _ = make_images_and_labels(2, 42)
z = latent_prior(2, torch.Generator().manual_seed(2026)).requires_grad_(True)
generated = model.generator(z)
assert generated.shape == images.shape == (2, 1, 8, 8)
assert torch.isfinite(generated).all() and generated.min() >= 0 and generated.max() <= 1.6
assert not torch.allclose(generated[0], generated[1])
assert model.discriminator(generated, z).shape == (2, 1)
generated.square().mean().backward()
assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
model.zero_grad(set_to_none=True)
z = z.detach()
ge_optimizer, d_optimizer = create_optimizers(model)
snapshot = lambda: {n: p.detach().clone() for n, p in model.named_parameters()}
before_d = snapshot()
d_loss = discriminator_step(model, images, z, d_optimizer)
for name, p in model.named_parameters():
    if name.startswith("discriminator."):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
        assert not torch.equal(before_d[name], p), name
    else:
        assert p.grad is None and torch.equal(before_d[name], p), name
print("Discriminator update isolation passed", flush=True)
model.zero_grad(set_to_none=True)
before_ge = snapshot()
ge_loss = generator_encoder_step(model, images, z, ge_optimizer)
gradient_norms = {}
for name, p in model.named_parameters():
    if name.startswith("discriminator."):
        assert p.grad is None and torch.equal(before_ge[name], p), name
        assert p.requires_grad
    else:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
        assert not torch.equal(before_ge[name], p), name
        gradient_norms[name] = p.grad.norm().item()
assert torch.isfinite(torch.tensor([d_loss, ge_loss])).all()
report = {"stage": 3, "status": "passed", "image_shape": list(generated.shape),
          "pair_input_size": 68, "discriminator_loss": d_loss, "generator_encoder_loss": ge_loss,
          "gradient_norms": gradient_norms,
          "checks": ["generated image shape/range", "latent sensitivity", "latent gradient",
                     "real/fake pair logits", "D-only update", "G/E-only update", "finite losses"]}
(root / "bigan_training_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
