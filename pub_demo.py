"""
PUB (Primitive Unified Bloc) demonstration
==========================================
Shows how to build and use PUBs with both StatevectorEstimator
and StatevectorSampler (local simulators, no IBM account needed).

Topics covered:
  1. Minimal PUB  — circuit only
  2. PUB with observables
  3. PUB with parameter values
  4. Broadcasting  — many parameter sets in one PUB
  5. Multiple PUBs in one .run() call
  6. Reading PubResult fields
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

SEP = "=" * 60

# ─────────────────────────────────────────────────────────────
# 1. MINIMAL SAMPLER PUB  (circuit only)
# ─────────────────────────────────────────────────────────────
print(SEP)
print("1. Minimal Sampler PUB — circuit only")
print(SEP)

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

sampler = StatevectorSampler()

pub    = (qc,)                  # PUB is a tuple: (circuit,)
result = sampler.run([pub]).result()

print(f"PUB structure : (circuit,)")
print(f"Circuit       : Bell state H + CX")
print(f"Counts        : {result[0].data.meas.get_counts()}")
print()

# ─────────────────────────────────────────────────────────────
# 2. SAMPLER PUB with shots override
# ─────────────────────────────────────────────────────────────
print(SEP)
print("2. Sampler PUB with shots override")
print(SEP)

pub_shots = (qc, None, 2048)    # (circuit, parameter_values, shots)
result    = sampler.run([pub_shots]).result()

print(f"PUB structure : (circuit, None, 2048)")
print(f"Shots used    : {result[0].metadata['shots']}")
print(f"Counts        : {result[0].data.meas.get_counts()}")
print()

# ─────────────────────────────────────────────────────────────
# 3. ESTIMATOR PUB — circuit + observable
# ─────────────────────────────────────────────────────────────
print(SEP)
print("3. Estimator PUB — circuit + single observable")
print(SEP)

qc_est = QuantumCircuit(2)
qc_est.h(0)
qc_est.cx(0, 1)

estimator  = StatevectorEstimator()
observable = SparsePauliOp("ZZ")   # ⟨ZZ⟩ should be +1 for Bell |Φ+⟩

pub    = (qc_est, observable)
result = estimator.run([pub]).result()

print(f"PUB structure      : (circuit, observable)")
print(f"Observable         : ZZ")
print(f"⟨ZZ⟩ (expect +1.0): {result[0].data.evs:.4f}")
print()

# ─────────────────────────────────────────────────────────────
# 4. ESTIMATOR PUB — parameterized circuit + parameter values
# ─────────────────────────────────────────────────────────────
print(SEP)
print("4. Estimator PUB — parameterized circuit + parameter values")
print(SEP)

theta = ParameterVector('θ', 2)

qc_param = QuantumCircuit(2)
qc_param.ry(theta[0], 0)
qc_param.ry(theta[1], 1)
qc_param.cx(0, 1)

observable = SparsePauliOp("ZZ")

# One specific parameter set
param_values = [np.pi / 4, np.pi / 3]

pub    = (qc_param, observable, param_values)
result = estimator.run([pub]).result()

print(f"PUB structure      : (circuit, observable, param_values)")
print(f"Parameters         : θ[0]={param_values[0]:.4f}  θ[1]={param_values[1]:.4f}")
print(f"⟨ZZ⟩               : {result[0].data.evs:.4f}")
print()

# ─────────────────────────────────────────────────────────────
# 5. BROADCASTING — many parameter sets, many observables
# ─────────────────────────────────────────────────────────────
print(SEP)
print("5. Broadcasting — shape (4,) params  x  shape (3,) observables")
print(SEP)

# 4 parameter sets, each with 2 values
param_array = np.array([
    [0,        0       ],
    [np.pi/4,  np.pi/4 ],
    [np.pi/2,  np.pi/2 ],
    [np.pi,    np.pi   ],
])  # shape (4, 2)

# 3 observables — shape (3, 1) so they broadcast against param_array shape (4,)
# resulting in evs shape (3, 4): one value per (observable, param_set)
observables = [[SparsePauliOp("ZZ")],
               [SparsePauliOp("XX")],
               [SparsePauliOp("YY")]]   # shape (3, 1)

pub    = (qc_param, observables, param_array)
result = estimator.run([pub]).result()

evs = result[0].data.evs   # shape (3, 4) — [obs, param_set]
print(f"param_array shape  : {param_array.shape}  → 4 parameter sets")
print(f"observables shape  : (3, 1)  → ZZ, XX, YY")
print(f"broadcast rule     : (3,1) x (4,) → result shape (3, 4)")
print(f"result evs shape   : {evs.shape}  — (observables, param_sets)")
print()
print(f"{'θ[0]/θ[1]':<16} {'⟨ZZ⟩':>8} {'⟨XX⟩':>8} {'⟨YY⟩':>8}")
print("-" * 44)
labels = ["0, 0", "π/4, π/4", "π/2, π/2", "π, π"]
for j, label in enumerate(labels):
    print(f"{label:<16} {evs[0,j]:>8.4f} {evs[1,j]:>8.4f} {evs[2,j]:>8.4f}")
print()

# ─────────────────────────────────────────────────────────────
# 6. MULTIPLE PUBs in one .run() call
# ─────────────────────────────────────────────────────────────
print(SEP)
print("6. Multiple PUBs in one .run() call")
print(SEP)

# PUB 0 — Bell state, ZZ observable
qc_bell = QuantumCircuit(2)
qc_bell.h(0)
qc_bell.cx(0, 1)
pub0 = (qc_bell, SparsePauliOp("ZZ"))

# PUB 1 — |+⟩ state, X observable (expect +1)
qc_plus = QuantumCircuit(1)
qc_plus.h(0)
pub1 = (qc_plus, SparsePauliOp("X"))

# PUB 2 — |1⟩ state, Z observable (expect -1)
qc_one = QuantumCircuit(1)
qc_one.x(0)
pub2 = (qc_one, SparsePauliOp("Z"))

result = estimator.run([pub0, pub1, pub2]).result()

print(f"Submitted 3 PUBs in one .run() call")
print(f"result[0] — Bell ⟨ZZ⟩  (expect +1.0): {result[0].data.evs:.4f}")
print(f"result[1] — |+⟩  ⟨X⟩   (expect +1.0): {result[1].data.evs:.4f}")
print(f"result[2] — |1⟩  ⟨Z⟩   (expect -1.0): {result[2].data.evs:.4f}")
print()

# ─────────────────────────────────────────────────────────────
# 7. PubResult fields
# ─────────────────────────────────────────────────────────────
print(SEP)
print("7. PubResult fields")
print(SEP)

pub    = (qc_bell, [SparsePauliOp("ZZ"), SparsePauliOp("ZI"), SparsePauliOp("IZ")])
result = estimator.run([pub]).result()
pub_result = result[0]

print(f"type(result[0])          : {type(pub_result).__name__}")
print(f"result[0].data.evs       : {pub_result.data.evs}   ← expectation values")
print(f"result[0].data.stds      : {pub_result.data.stds}  ← standard deviations")
print(f"result[0].metadata       : {pub_result.metadata}")
print()
print(SEP)
print("PUB structure summary")
print(SEP)
rows = [
    ("Sampler",   "(circuit,)",                               "minimal"),
    ("Sampler",   "(circuit, None, shots)",                   "override shots"),
    ("Estimator", "(circuit, observables)",                   "minimal"),
    ("Estimator", "(circuit, observables, param_values)",     "with parameters"),
    ("Estimator", "(circuit, observables, param_array, prec)","full PUB"),
]
print(f"  {'Primitive':<12} {'PUB tuple':<44} {'Notes'}")
print("  " + "-" * 70)
for prim, tup, note in rows:
    print(f"  {prim:<12} {tup:<44} {note}")
print(SEP)
