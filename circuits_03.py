from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram
from qiskit.visualization import (plot_bloch_multivector, plot_state_qsphere,
                                   plot_state_city, plot_state_paulivec,
                                   plot_state_hinton)
import matplotlib.pyplot as plt
import qiskit
print("Qiskit version:", qiskit.__version__)
import importlib.metadata
packages = ["qiskit", "qiskit-ibm-runtime", "qiskit-aer"]
for pkg in packages:
    try:
        print(f"  {pkg}: {importlib.metadata.version(pkg)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"  {pkg}: not installed")

# HXH = Z demonstration
qc = QuantumCircuit(2)
qc.h(0)
qc.x(1)
state = Statevector(qc)

print("Statevector:", state)
print("Circuit (text):")
print(qc.draw('text'))

# Circuit diagram (mpl)
qc.draw('mpl').suptitle("Circuit")

# Measure the output
qc_meas = qc.copy()
qc_meas.measure_all()
sampler = StatevectorSampler()
result = sampler.run([qc_meas], shots=1024).result()
counts = result[0].data.meas.get_counts()
print("Counts:", counts)
plot_histogram(counts).suptitle("Measurement Histogram")

# State plots
plot_state_qsphere(state).suptitle("Q-sphere")
plot_bloch_multivector(state).suptitle("Bloch Multivector")
plot_state_city(state).suptitle("City")
plot_state_paulivec(state).suptitle("Pauli Vector")
plot_state_hinton(state).suptitle("Hinton")

from qiskit.circuit.library import real_amplitudes

ansatz = real_amplitudes(5, entanglement="pairwise", reps=2)
ansatz.draw("mpl")

plt.show()
