"""Package the validated stages as an IDE-friendly notebook."""
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parent
n = nbformat.v4.new_notebook()
n.metadata["kernelspec"] = {"display_name": "DOUBT_QC", "language": "python", "name": "python3"}
n.cells = [
    nbformat.v4.new_markdown_cell("""# Quantum BiGAN — structured synthetic worlds

The original `GAN_01.ipynb` and stage-one `GAN_01_latent4.ipynb` are preserved.
Implementation is in `quantum_bigan.py`; this notebook validates stages in order
and runs a reproducible experiment in the existing DOUBT_QC environment.

The dataset is reproduced exactly from `I_JEPA_05.ipynb`, cell 2: grayscale
images of shape `(1,8,8)` with `image=a*x+b*y`, `a,b` in `[0.2,0.8)`.
Training has 256 images (seed 42); testing has 128 (seed 123). Intensities can
exceed one, so no clipping or per-image normalization is applied.

The latent prior is Uniform(-pi,pi)^4. The generator retains the stage-one
six-qubit SamplerQNN and 84 quantum weights, adding a learned pixel decoder.
It consumes measured probabilities as features, rather than sampling an image
from a bitstring. The encoder uses a learned 64-pixel-to-four-angle projection,
a four-qubit EstimatorQNN with 24 weights, and four Pauli-Z expectations scaled
by pi. Both QNNs retain TorchConnector with input gradients enabled.

The 20-unit classical discriminator receives 64 image pixels and four latents.
It scores `(x,E(x))` as real and `(G(z),z)` as generated. D minimizes balanced
BCE; G/E use the non-saturating objective with both labels flipped. No
reconstruction or factor supervision is added to adversarial training.

References: [original Qiskit qGAN tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/04_torch_qgan.html)
and [TorchConnector tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/05_torch_connector.html)."""),
    nbformat.v4.new_code_cell("""import json
import subprocess
import sys
from pathlib import Path

import torch
from quantum_bigan import make_images_and_labels, verify_dataset_source

ROOT = Path.cwd()
assert (ROOT / 'quantum_bigan.py').is_file(), 'Open this notebook from DOUBT_QC'
print('Interpreter:', sys.executable)
print(json.dumps(verify_dataset_source(), indent=2))"""),
    nbformat.v4.new_code_cell("""train_images, train_labels = make_images_and_labels(256, 42)
test_images, test_labels = make_images_and_labels(128, 123)
print('Train/test shapes:', tuple(train_images.shape), tuple(test_images.shape))
print('Training intensity range:', train_images.min().item(), train_images.max().item())"""),
    nbformat.v4.new_markdown_cell("""## Validate each stage before training

Each check runs with this kernel's interpreter. Failure stops execution before
training. These checks also confirm that the original notebooks remain unchanged."""),
    nbformat.v4.new_code_cell("""for script in ['validate_latent_generator.py', 'validate_quantum_encoder.py', 'validate_bigan_training.py']:
    completed = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT,
                               capture_output=True, text=True)
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    completed.check_returncode()"""),
    nbformat.v4.new_markdown_cell("""## Development training run

Defaults use two epochs on a fixed 32-image subset of the training split, keeping
CPU validation practical. Evaluation uses all 128 test images. The linear factor
probe is fitted on all 256 training images only; factor labels never enter the
adversarial losses. Increase TRAIN_COUNT to 256 and EPOCHS for a full experiment.
This short run demonstrates execution and gradient flow, not convergence.
The generator uses 1024 shots and a fixed simulator seed for reproducibility."""),
    nbformat.v4.new_code_cell("""EPOCHS = 2
TRAIN_COUNT = 32
BATCH_SIZE = 4
OUTPUT_DIR = ROOT / 'bigan_results'

command = [sys.executable, str(ROOT / 'run_quantum_bigan.py'),
           '--epochs', str(EPOCHS), '--train-count', str(TRAIN_COUNT),
           '--batch-size', str(BATCH_SIZE), '--output-dir', str(OUTPUT_DIR)]
completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
print(completed.stdout)
if completed.stderr:
    print(completed.stderr)
completed.check_returncode()"""),
    nbformat.v4.new_code_cell("""report = json.loads((OUTPUT_DIR / 'evaluation.json').read_text())
print(json.dumps({'initial': report['initial'], 'final': report['final'],
                  'history': report['history']}, indent=2))
from IPython.display import Image, display
display(Image(filename=str(OUTPUT_DIR / 'examples.png')))"""),
    nbformat.v4.new_markdown_cell("""## Reading the evaluation

The encoder probe predicts the known `(a,b)` factors; compare held-out MSE and
R² with the untrained encoder, raw pixels, and training-mean labels. Raw pixels
already contain these factors linearly, so this dataset cannot by itself
demonstrate a quantum advantage.

Image RBF MMD² compares generated and held-out image distributions using a
bandwidth chosen from training data only. Reconstruction MSE is diagnostic:
the BiGAN objective does not directly optimize it. Pixel variance and the
residual from the known planar image structure help detect collapse and
off-manifold outputs. Generated images are independent samples and are not
paired with the real images displayed above them. Reconstructions correspond
to those real images. Finite-shot generator probabilities introduce simulation
noise; repeat longer experiments across seeds before drawing quality conclusions.

Results and weights are saved separately under `bigan_results`, leaving the
original qGAN and JEPA artifacts intact."""),
]
nbformat.write(n, root / "GAN_02_BiGAN.ipynb")
print("Created GAN_02_BiGAN.ipynb")
