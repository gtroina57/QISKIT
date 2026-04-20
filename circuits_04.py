# --- Imports ---
# Core Qiskit classes for building and simulating quantum circuits
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler

# Visualization tools for histograms and quantum state representations
from qiskit.visualization import plot_histogram
from qiskit.visualization import (plot_bloch_multivector, plot_state_qsphere,
                                   plot_state_city, plot_state_paulivec,
                                   plot_state_hinton)
import matplotlib.pyplot as plt

# --- Version check: print installed Qiskit package versions ---
import qiskit
print("Qiskit version:", qiskit.__version__)

import importlib.metadata
packages = ["qiskit", "qiskit-ibm-runtime", "qiskit-aer"]
for pkg in packages:
    try:
        print(f"  {pkg}: {importlib.metadata.version(pkg)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"  {pkg}: not installed")

# --- Quantum Register and Ancilla Register Example ---

from qiskit.visualization import plot_error_map
from qiskit.providers.fake_provider import GenericBackendV2

from qiskit.circuit import QuantumRegister, ClassicalRegister

from qiskit.circuit import Parameter

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import generate_preset_pass_manager

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2

backend = GenericBackendV2(15)


ghz = QuantumCircuit(15)
ghz.h(0)
ghz.cx(0, range(1, 15))

depths = []
for seed in range(100):
    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend,
        layout_method="trivial",  # Fixed layout mapped in circuit order
        seed_transpiler=seed,  # For reproducible results
    )
    depths.append(pass_manager.run(ghz).depth())

plt.figure(figsize=(8, 6))
plt.hist(depths, align="left", color="#AC557C")
plt.xlabel("Depth", fontsize=14)
plt.ylabel("Counts", fontsize=14)

ghz.draw("mpl", idle_wires=False)

from qiskit.visualization import plot_circuit_layout

# Plot the hardware graph and indicate which hardware qubits were chosen to run the circuit
transpiled_circ = pass_manager.run(ghz)
plot_circuit_layout(transpiled_circ, backend)

plt.show()
