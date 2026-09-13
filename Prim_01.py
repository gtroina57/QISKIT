# 1. Import the local simulator backend
from qiskit_aer import AerSimulator
# We'll start with an ideal (noiseless) simulator.
backend = AerSimulator()

print(f"Using local backend: {backend.name}")

from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp
import numpy as np

# To make this runnable and easy to inspect, let's use a smaller 5-qubit circuit.
num_qubits = 5
entanglement = [(0, 1), (1, 2), (2, 3), (3, 4)]
observable = SparsePauliOp.from_sparse_list(
    [("ZZ", [i, j], 0.5) for i, j in entanglement],
    num_qubits=num_qubits,
)
circuit = qaoa_ansatz(observable, reps=2)

# The parameter values now need to match the smaller circuit's parameter count
param_values = np.random.rand(circuit.num_parameters)

print(f">>> Circuit has {circuit.num_parameters} parameters.")
print(f">>> Observable Paulis: {observable.paulis}")

from qiskit.transpiler import generate_preset_pass_manager

# Transpiling the circuit for the backend's basis gates is good practice,
# though often optional for an ideal AerSimulator.
pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
isa_circuit = pm.run(circuit)
isa_observable = observable.apply_layout(isa_circuit.layout)
print(f">>> Circuit ops (ISA-compliant): {isa_circuit.count_ops()}")

# 2. Import the BackendEstimatorV2, which works with any local backend
from qiskit.primitives import BackendEstimatorV2 as Estimator

# Instantiate the Estimator with our local AerSimulator backend
estimator = Estimator(backend=backend)

# The run call takes a list of Primitives Unified Blocs (PUBs).
# Each PUB is a tuple of (circuit, observable, [parameter_values]).
job = estimator.run([(isa_circuit, isa_observable, param_values)])
print(f">>> Job ID: {job.job_id()}")
print(f">>> Job Status: {job.status()}")

# 3. Crucially, let's actually look at the results!
result = job.result()
print(f"\n>>> Full Result Object: {result}")

# The result for the first PUB is at index 0
pub_result = result[0]
print(f"\n  > Expectation value (EV): {pub_result.data.evs}")
print(f"  > Standard deviation: {pub_result.data.stds}")
print(f"  > Metadata: {pub_result.metadata}")

