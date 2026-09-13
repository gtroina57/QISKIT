from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import matplotlib.pyplot as plt

n_qubits = 3

# Hadamard on every qubit — produces uniform superposition over all 2^n bitstrings
circuit = QuantumCircuit(n_qubits)
circuit.h(range(n_qubits))
circuit.measure_all()

print("Circuit:")
print(circuit.draw('text'))

# Connect to IBM and pick least busy backend
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)
print(f"\nBackend: {backend.name}")

# Transpile
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(circuit)

# Run on hardware
sampler = Sampler(backend, options={"default_shots": 4096})
job = sampler.run([isa_circuit])
print(f"Job ID: {job.job_id()}")

result = job.result()
pub_result = result[0]

counts = pub_result.data.meas.get_counts()
print(f"\nCounts: {counts}")

# Plot distribution
plot_histogram(counts, title=f"Hadamard on all {n_qubits} qubits — expected uniform distribution")
plt.show()
