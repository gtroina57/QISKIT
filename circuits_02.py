from math import sqrt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt

qc = QuantumCircuit(1, 1)
state_vector = [1/sqrt(2), 1j/sqrt(2)] # [0.70710678118, 0.70710678118]
qc.initialize(state_vector, 0)

state = Statevector(qc)
figure = plot_bloch_multivector(state)
plt.show()
