from qiskit.circuit import (
    Parameter,
    QuantumCircuit,
    ClassicalRegister,
    QuantumRegister,
)
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives.containers import BitArray
from qiskit.primitives import StatevectorEstimator


import numpy as np

# Define a circuit with two parameters.
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(Parameter("a"), 0)
circuit.rz(Parameter("b"), 0)
circuit.cx(0, 1)
circuit.h(0)

# Transpile the circuit without providing a backend
pm = generate_preset_pass_manager(optimization_level=1)
transpiled_circuit = pm.run(circuit)
layout = transpiled_circuit.layout

# Now define a sweep over parameter values, the last axis of dimension 2 is
# for the two parameters "a" and "b"
params = np.vstack(
    [
        np.linspace(-np.pi, np.pi, 10),
        np.linspace(-4 * np.pi, 4 * np.pi, 10),
    ]
).T

# Define three observables. The inner length-1 lists cause this array of
# observables to have shape (3, 1), rather than shape (3,) if they were
# omitted.
observables = [
    [SparsePauliOp(["XX", "IY"], [0.5, 0.5])],
    [SparsePauliOp("XX")],
    [SparsePauliOp("IY")],
]
# Apply the same layout as the transpiled circuit.
observables = [
    [observable.apply_layout(layout) for observable in observable_set]
    for observable_set in observables
]

# Estimate the expectation value for all 300 combinations of observables
# and parameter values, where the pub result will have shape (3, 100).
#
# This shape is due to our array of parameter bindings having shape
# (100, 2), combined with our array of observables having shape (3, 1).
estimator = StatevectorEstimator()
estimator_pub = (transpiled_circuit, observables, params)

# Run the transpiled circuit
# using the set of parameters and observables.

job = estimator.run([estimator_pub])
result = job.result()
# Access the PUB result (index 0 because we passed one PUB)
pub_result = result[0]

# The data contains the expectation values and standard deviations
evs = pub_result.data.evs       # expectation values, shape (3, 10)
stds = pub_result.data.stds     # standard deviations, shape (3, 10)

print("Expectation values shape:", evs.shape)
print()

observable_names = ["0.5·XX + 0.5·IY", "XX", "IY"]

for i, name in enumerate(observable_names):
    print(f"Observable: {name}")
    for j in range(10):
        print(f"  (a={params[j,0]:.3f}, b={params[j,1]:.3f})  →  ⟨O⟩ = {evs[i,j]:.4f} ± {stds[i,j]:.4f}")
    print()

print(result)

##plt.show()