"""Create the minimal stage-one variant without modifying GAN_01.ipynb."""
import copy
import json
from pathlib import Path

root = Path(__file__).resolve().parent
original = json.loads((root / "GAN_01.ipynb").read_text(encoding="utf-8"))
notebook = copy.deepcopy(original)

def replace(index, old, new):
    source = "".join(notebook["cells"][index]["source"])
    assert old in source, (index, old)
    notebook["cells"][index]["source"] = source.replace(old, new).splitlines(keepends=True)

replace(1, "num_dim = 2", "latent_dim = 4\nlatent_batch_size = 4\n\nnum_dim = 2")
replace(4, "from qiskit.circuit.library import efficient_su2", "from qiskit.circuit.library import efficient_su2\nfrom qiskit.circuit import ParameterVector")
replace(4, "qc.h(qc.qubits)", "qc.h(qc.qubits)\n\n# Four independent latent inputs; the six-qubit output grid is unchanged.\nlatent_params = ParameterVector(\"z\", latent_dim)\nfor qubit, angle in enumerate(latent_params):\n    qc.ry(angle, qubit)")
replace(8, "input_params=[],", "input_params=list(latent_params),")
replace(8, "weight_params=qc.parameters,", "weight_params=list(ansatz.parameters),\n        input_gradients=True,")
replace(8, "random(qc.num_parameters)", "random(ansatz.num_parameters)")
replace(14, "gen_dist = generator(torch.tensor([])).reshape(-1, 1)", "z = (2 * torch.rand(latent_batch_size, latent_dim) - 1) * torch.pi\n    gen_dist = generator(z).mean(dim=0).reshape(-1, 1)")
replace(15, "generated_probabilities = generator().numpy()", "# Estimate the marginal distribution over the same uniform latent prior.\n    z_eval = (2 * torch.rand(32, latent_dim) - 1) * torch.pi\n    generated_probabilities = generator(z_eval).mean(dim=0).numpy()")

intro = """# Quantum qGAN — stage 1: four-dimensional latent input

This separate notebook preserves `GAN_01.ipynb`. It retains the tutorial's
six-qubit, 64-outcome distribution generator, `SamplerQNN`, `TorchConnector`,
84 trainable ansatz parameters, classical discriminator, and weighted losses.
Four `Ry(z[i])` rotations after the Hadamards encode a uniform latent prior
on [-pi, pi]^4. Latent parameters are explicitly separated from weights.
Training averages four conditional distributions; final plots average 32.
Input gradients are enabled for future encoder integration.

This stage still models the original two-dimensional Gaussian grid. The image
decoder, four-qubit encoder, pair discriminator, and BiGAN objective belong to
later stages after this change is validated and the dataset is identified.

Reference: [Qiskit PyTorch qGAN tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/04_torch_qgan.html).
Run `validate_latent_generator.py` with the DOUBT_QC interpreter for a short
forward/backward and two-epoch training check without running all 50 epochs.
"""
notebook["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)})
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
(root / "GAN_01_latent4.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("Created GAN_01_latent4.ipynb")
