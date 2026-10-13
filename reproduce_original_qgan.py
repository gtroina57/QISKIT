"""Execute original tutorial cells unchanged and retain CDF evidence separately."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="qiskit231_unseeded")
    args = parser.parse_args()
    out = ROOT / "tutorial_reproduction" / args.label
    out.mkdir(parents=True, exist_ok=True)
    original = ROOT / "GAN_01.ipynb"
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    n = nbformat.read(original, as_version=4)
    sources = [c.source for c in n.cells if c.cell_type == "code"]
    for c in n.cells:
        if c.cell_type == "code":
            c.execution_count = None
            c.outputs = []
    n.cells.insert(0, nbformat.v4.new_markdown_cell(
        "# Original Qiskit qGAN reproduction\n\n"
        "All original code cells from `GAN_01.ipynb` are executed unchanged, including "
        "`Sampler()` without a seed, 50 epochs, losses, and plotting. "
        "Only this introduction and a final diagnostic cell are added. "
        "The source notebook is preserved. Results are one stochastic run, "
        "not an exact recreation of the published measurement draws.\n\n"
        "Reference: https://qiskit-community.github.io/qiskit-machine-learning/tutorials/04_torch_qgan.html\n\n"
        "Copyright IBM 2017, 2026. Original tutorial code is licensed under Apache-2.0."
    ))
    diagnostic = '''
import json, sys, platform, hashlib
from importlib.metadata import version
from pathlib import Path
from qiskit.quantum_info import Statevector

out = Path(OUTPUT_DIRECTORY)
p_sampled = np.asarray(generated_probabilities, dtype=float)
p_exact = Statevector.from_instruction(qc.assign_parameters(
    dict(zip(qc.parameters, generator.weight.detach().numpy())))).probabilities()
target = np.asarray(prob_data, dtype=float)

def metrics(p):
    difference = np.cumsum(target) - np.cumsum(p)
    # The tutorial flattens then cumulatively sums. A joint 2D CDF is different.
    joint_difference = target.reshape(8, 8).cumsum(0).cumsum(1) - p.reshape(8, 8).cumsum(0).cumsum(1)
    return {
        "kl_generated_to_target": float(entropy(p, target)),
        "total_variation": float(np.abs(p - target).sum() / 2),
        "tutorial_cdf_difference_min": float(difference.min()),
        "tutorial_cdf_difference_max": float(difference.max()),
        "tutorial_cdf_max_absolute_difference": float(np.abs(difference).max()),
        "joint_2d_cdf_max_absolute_difference": float(np.abs(joint_difference).max()),
    }

assert np.isclose(p_sampled.sum(), 1) and np.isclose(p_exact.sum(), 1)
assert np.isfinite(generator_loss_values).all() and np.isfinite(discriminator_loss_values).all()
report = {
    "source_notebook_sha256": SOURCE_DIGEST, "original_code_cells_unchanged": True,
    "sampler_seed": sampler.seed, "shots": sampler.default_shots, "epochs": n_epochs,
    "versions": {p: version(p) for p in ("qiskit", "qiskit-machine-learning", "torch", "numpy", "scipy")},
    "python": sys.version, "os": platform.platform(),
    "sampled": metrics(p_sampled), "noiseless": metrics(p_exact),
    "sampled_probabilities": p_sampled.tolist(), "noiseless_probabilities": p_exact.tolist(),
    "target_probabilities": target.tolist(),
    "generator_losses": generator_loss_values, "discriminator_losses": discriminator_loss_values,
    "training_batch_kl": entropy_values, "training_seconds": elapsed,
}
(out / "results.json").write_text(json.dumps(report, indent=2) + "\\n", encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, indent=2))
'''
    diagnostic = diagnostic.replace("OUTPUT_DIRECTORY", repr(str(out))).replace("SOURCE_DIGEST", repr(digest))
    n.cells.append(nbformat.v4.new_code_cell(diagnostic))
    assert [c.source for c in n.cells if c.cell_type == "code"][:-1] == sources
    manager = KernelManager(kernel_name="python3")
    env = dict(os.environ)
    env["IPYTHONDIR"] = str(out / "ipython")
    manager._kernel_spec = KernelSpec(
        argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        display_name="DOUBT_QC", language="python", env={"IPYTHONDIR": env["IPYTHONDIR"]})
    client = NotebookClient(n, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}})
    client.km = manager
    print(f"Executing unchanged original cells: {out}", flush=True)
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        nbformat.write(n, out / "GAN_01_reproduced.ipynb")
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
    for index, cell in enumerate(n.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.outputs:
            data = output.get("data", {})
            if "image/png" in data:
                (out / f"cell_{index:02d}.png").write_bytes(base64.b64decode(data["image/png"]))
    report = json.loads((out / "results.json").read_text(encoding="utf-8"))
    print(json.dumps({"versions": report["versions"], "sampler_seed": report["sampler_seed"],
                      "last_training_kl": report["training_batch_kl"][-1],
                      "sampled": report["sampled"], "noiseless": report["noiseless"]}, indent=2), flush=True)
    print("Original notebook unchanged; completed reproduction saved.", flush=True)


if __name__ == "__main__":
    main()
