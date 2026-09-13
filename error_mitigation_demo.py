"""
Error Mitigation Demo
=====================
Demonstrates four mitigation techniques on a noisy simulator:

  1. Baseline        — noisy, no mitigation
  2. ZNE             — Zero Noise Extrapolation (gate folding)
  3. Meas. Mitigation— calibration matrix inversion
  4. Pauli Twirling  — symmetrise noise → depolarising channel
  5. ZNE + Twirling  — combined

Target: ⟨ZZ⟩ on a Bell state |Φ+⟩ — ideal value is exactly +1.0
Noise:  depolarising on gates + readout errors
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

SHOTS  = 8192
SEP    = "=" * 60

# ─────────────────────────────────────────────────────────────
# 1. Noise model
# ─────────────────────────────────────────────────────────────
noise_model = NoiseModel()

# Gate errors
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['h', 'x', 'rz', 'sx'])
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.03, 2), ['cx'])

# Readout errors  (3% chance of flipping each bit)
ro_error = ReadoutError([[0.97, 0.03], [0.03, 0.97]])
noise_model.add_all_qubit_readout_error(ro_error)

backend_noisy = AerSimulator(noise_model=noise_model)
backend_ideal = AerSimulator()   # noiseless

# ─────────────────────────────────────────────────────────────
# 2. Target circuit — Bell state ⟨ZZ⟩ = +1.0
# ─────────────────────────────────────────────────────────────
def bell_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def get_expectation_ZZ(counts, shots):
    """Compute ⟨ZZ⟩ from raw counts."""
    ev = 0
    for bitstring, count in counts.items():
        z0 = 1 - 2 * int(bitstring[-1])   # qubit 0
        z1 = 1 - 2 * int(bitstring[-2])   # qubit 1
        ev += (z0 * z1) * count
    return ev / shots

# Ideal value (statevector)
sv    = Statevector(bell_circuit())
ideal = sv.expectation_value(SparsePauliOp("ZZ")).real
print(f"Ideal ⟨ZZ⟩ (statevector): {ideal:.4f}")

# ─────────────────────────────────────────────────────────────
# 3. Baseline — noisy, no mitigation
# ─────────────────────────────────────────────────────────────
print(SEP)
print("1. Baseline — noisy, no mitigation")
print(SEP)

qc = bell_circuit()
qc.measure_all()
t_qc     = transpile(qc, backend_noisy, optimization_level=0)
counts   = backend_noisy.run(t_qc, shots=SHOTS).result().get_counts()
ev_noisy = get_expectation_ZZ(counts, SHOTS)
print(f"Counts : {dict(sorted(counts.items()))}")
print(f"⟨ZZ⟩   : {ev_noisy:.4f}  (ideal = {ideal:.4f})")
print()

# ─────────────────────────────────────────────────────────────
# 4. ZNE — Zero Noise Extrapolation via gate folding
# ─────────────────────────────────────────────────────────────
print(SEP)
print("2. ZNE — gate folding at scales 1, 3, 5")
print(SEP)

def fold_gates(qc, scale):
    """
    Scale noise by replacing each CX with CX·CX†·CX (scale=3)
    or CX·CX†·CX·CX†·CX (scale=5).  Scale=1 = no folding.
    """
    if scale == 1:
        return qc
    folds = (scale - 1) // 2   # number of (CX†·CX) pairs to append
    folded = QuantumCircuit(qc.num_qubits)
    for instr in qc.data:
        folded.append(instr)
        if instr.operation.name == 'cx':
            qargs = instr.qubits
            for _ in range(folds):
                folded.cx(*[qc.find_bit(q).index for q in qargs])
                folded.cx(*[qc.find_bit(q).index for q in qargs])
    return folded

scales    = [1, 3, 5]
zne_evs   = []
base_qc   = bell_circuit()

for scale in scales:
    folded = fold_gates(base_qc.copy(), scale)
    folded.measure_all()
    t = transpile(folded, backend_noisy, optimization_level=0)
    c = backend_noisy.run(t, shots=SHOTS).result().get_counts()
    ev = get_expectation_ZZ(c, SHOTS)
    zne_evs.append(ev)
    print(f"  Scale {scale}: ⟨ZZ⟩ = {ev:.4f}")

# Linear extrapolation to noise = 0
coeffs    = np.polyfit(scales, zne_evs, 1)
ev_zne    = np.polyval(coeffs, 0)
print(f"\n  Extrapolated to noise=0: ⟨ZZ⟩ = {ev_zne:.4f}")
print()

# ─────────────────────────────────────────────────────────────
# 5. Measurement Error Mitigation — calibration matrix
# ─────────────────────────────────────────────────────────────
print(SEP)
print("3. Measurement Error Mitigation — calibration matrix")
print(SEP)

def build_calibration_matrix(backend, num_qubits, shots):
    """Prepare each basis state and measure to build the A matrix."""
    n_states = 2 ** num_qubits
    cal_matrix = np.zeros((n_states, n_states))

    for i in range(n_states):
        qc_cal = QuantumCircuit(num_qubits)
        bitstring = format(i, f'0{num_qubits}b')
        for j, bit in enumerate(reversed(bitstring)):
            if bit == '1':
                qc_cal.x(j)
        qc_cal.measure_all()
        t = transpile(qc_cal, backend, optimization_level=0)
        c = backend.run(t, shots=shots).result().get_counts()
        for j in range(n_states):
            key = format(j, f'0{num_qubits}b')
            cal_matrix[j, i] = c.get(key, 0) / shots

    return cal_matrix

print("  Building calibration matrix...")
cal_matrix = build_calibration_matrix(backend_noisy, 2, SHOTS)
print(f"  Calibration matrix:\n{np.round(cal_matrix, 3)}")

# Apply mitigation: solve A·p_mitigated = p_noisy
qc_m   = bell_circuit()
qc_m.measure_all()
t_m    = transpile(qc_m, backend_noisy, optimization_level=0)
counts_m = backend_noisy.run(t_m, shots=SHOTS).result().get_counts()

# Convert counts to probability vector
p_noisy = np.array([counts_m.get(format(i, '02b'), 0) for i in range(4)]) / SHOTS

# Least-squares inversion
p_mitigated, _ = np.linalg.lstsq(cal_matrix, p_noisy, rcond=None)[:2]
p_mitigated    = np.clip(p_mitigated, 0, None)
p_mitigated   /= p_mitigated.sum()

# Compute ⟨ZZ⟩ from mitigated probabilities
ev_meas_mit = 0
for i in range(4):
    bits = format(i, '02b')
    z0   = 1 - 2 * int(bits[1])
    z1   = 1 - 2 * int(bits[0])
    ev_meas_mit += z0 * z1 * p_mitigated[i]

print(f"\n  Noisy  ⟨ZZ⟩ : {get_expectation_ZZ(counts_m, SHOTS):.4f}")
print(f"  Mitigated ⟨ZZ⟩ : {ev_meas_mit:.4f}")
print()

# ─────────────────────────────────────────────────────────────
# 6. Pauli Twirling
# ─────────────────────────────────────────────────────────────
print(SEP)
print("4. Pauli Twirling — average over random Pauli insertions")
print(SEP)

# Pauli pairs (P1, P2) that preserve CX up to known corrections
# CX maps Paulis as: IX→IX, XI→XX, IZ→ZZ, ZI→ZI (stabiliser tableau)
# For each input Pauli P1⊗P2, the correction Pauli Q1⊗Q2 satisfies:
# Q·CX·P = CX  (up to phase)
TWIRL_TABLE = [
    ('II', 'II'), ('IX', 'IX'), ('IY', 'ZY'), ('IZ', 'ZZ'),
    ('XI', 'XX'), ('XX', 'XI'), ('XY', 'YZ'), ('XZ', 'YI'),
    ('YI', 'YX'), ('YX', 'YI'), ('YY', 'XZ'), ('YZ', 'XI'),
    ('ZI', 'ZI'), ('ZX', 'ZX'), ('ZY', 'IY'), ('ZZ', 'IZ'),
]

PAULI_GATE = {'I': None, 'X': 'x', 'Y': 'y', 'Z': 'z'}

def apply_pauli(qc, pauli, qubit):
    if pauli != 'I':
        getattr(qc, PAULI_GATE[pauli])(qubit)

def twirled_bell(rng):
    pre, post = TWIRL_TABLE[rng.integers(len(TWIRL_TABLE))]
    qc = QuantumCircuit(2)
    qc.h(0)
    apply_pauli(qc, pre[0],  0)
    apply_pauli(qc, pre[1],  1)
    qc.cx(0, 1)
    apply_pauli(qc, post[0], 0)
    apply_pauli(qc, post[1], 1)
    return qc

rng          = np.random.default_rng(42)
N_TWIRL      = 32
twirl_evs    = []

for _ in range(N_TWIRL):
    qc_t = twirled_bell(rng)
    qc_t.measure_all()
    t    = transpile(qc_t, backend_noisy, optimization_level=0)
    c    = backend_noisy.run(t, shots=SHOTS // N_TWIRL).result().get_counts()
    twirl_evs.append(get_expectation_ZZ(c, SHOTS // N_TWIRL))

ev_twirl = np.mean(twirl_evs)
print(f"  Averaged over {N_TWIRL} random twirls")
print(f"  Twirled ⟨ZZ⟩ : {ev_twirl:.4f}")
print()

# ─────────────────────────────────────────────────────────────
# 7. ZNE + Twirling combined
# ─────────────────────────────────────────────────────────────
print(SEP)
print("5. ZNE + Twirling — combined mitigation")
print(SEP)

combined_evs = []
for scale in scales:
    scale_evs = []
    for _ in range(N_TWIRL):
        qc_c = twirled_bell(rng)
        qc_c = fold_gates(qc_c.copy(), scale)
        qc_c.measure_all()
        t    = transpile(qc_c, backend_noisy, optimization_level=0)
        c    = backend_noisy.run(t, shots=SHOTS // N_TWIRL).result().get_counts()
        scale_evs.append(get_expectation_ZZ(c, SHOTS // N_TWIRL))
    combined_evs.append(np.mean(scale_evs))
    print(f"  Scale {scale}: ⟨ZZ⟩ = {combined_evs[-1]:.4f}")

coeffs_c  = np.polyfit(scales, combined_evs, 1)
ev_combined = np.polyval(coeffs_c, 0)
print(f"\n  ZNE+Twirling extrapolated: ⟨ZZ⟩ = {ev_combined:.4f}")
print()

# ─────────────────────────────────────────────────────────────
# 8. Summary table
# ─────────────────────────────────────────────────────────────
print(SEP)
print("Summary — ⟨ZZ⟩ comparison")
print(SEP)
results = {
    "Ideal (noiseless)":         ideal,
    "Noisy baseline":            ev_noisy,
    "ZNE":                       ev_zne,
    "Measurement mitigation":    ev_meas_mit,
    "Pauli twirling":            ev_twirl,
    "ZNE + Twirling":            ev_combined,
}
for label, val in results.items():
    error = abs(val - ideal)
    bar   = "█" * int(error * 50)
    print(f"  {label:<30} {val:>7.4f}   error={error:.4f}  {bar}")
print()

# ─────────────────────────────────────────────────────────────
# 9. Bar chart
# ─────────────────────────────────────────────────────────────
labels = list(results.keys())
values = list(results.values())
errors = [abs(v - ideal) for v in values]
colors = ["green", "red", "steelblue", "orange", "purple", "darkcyan"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Error Mitigation Demo — ⟨ZZ⟩ on Bell State  (ideal = 1.0)",
             fontsize=13, fontweight="bold")

# Left: expectation values
bars = ax1.bar(labels, values, color=colors)
ax1.axhline(ideal, color='black', linestyle='--', linewidth=1.5, label='Ideal')
ax1.set_ylim(0, 1.15)
ax1.set_ylabel("⟨ZZ⟩")
ax1.set_title("Expectation value per technique")
ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
ax1.legend()
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.3f}", ha='center', va='bottom', fontsize=8)

# Right: absolute error from ideal
ax2.bar(labels, errors, color=colors)
ax2.set_ylabel("|error| from ideal")
ax2.set_title("Absolute error from ideal (lower = better)")
ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
for i, (bar, err) in enumerate(zip(ax2.patches, errors)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{err:.3f}", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("error_mitigation_demo.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: error_mitigation_demo.png")

# ─────────────────────────────────────────────────────────────
# 10. ZNE extrapolation plot
# ─────────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(8, 5))
x_fit = np.linspace(0, 5.5, 100)

ax.plot(scales, zne_evs, 'o-',  color='steelblue', label='ZNE only',         ms=8)
ax.plot(scales, combined_evs, 's--', color='darkcyan',  label='ZNE + Twirling', ms=8)
ax.plot(x_fit, np.polyval(coeffs,   x_fit), ':', color='steelblue', alpha=0.6)
ax.plot(x_fit, np.polyval(coeffs_c, x_fit), ':', color='darkcyan',  alpha=0.6)
ax.axhline(ideal,    color='black', linestyle='-',  linewidth=1.5, label=f'Ideal = {ideal:.3f}')
ax.axvline(0,        color='grey',  linestyle='--', linewidth=1,   label='Extrapolation point')
ax.scatter([0], [ev_zne],      color='steelblue', s=120, zorder=5,
           label=f'ZNE extrapolated = {ev_zne:.3f}')
ax.scatter([0], [ev_combined], color='darkcyan',  s=120, zorder=5,
           label=f'ZNE+Twirl extrapolated = {ev_combined:.3f}')

ax.set_xlabel("Noise scale factor")
ax.set_ylabel("⟨ZZ⟩")
ax.set_title("ZNE — linear extrapolation to zero noise", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("error_mitigation_zne.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: error_mitigation_zne.png")
