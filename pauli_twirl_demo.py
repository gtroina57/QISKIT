"""
Pauli Twirling Demo
===================
Implements twirling manually (no PauliTwirl import needed).
Shows a circuit before and after twirling with 3 different random seeds.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

# ── Twirl table for CX ───────────────────────────────────────────────────────
# Each entry: (input_pauli, correction_pauli)
# Satisfies: correction · CX · input = CX  (up to global phase)
# Format: "XY" means X on control, Y on target
TWIRL_TABLE = [
    ('II', 'II'), ('IX', 'IX'), ('IY', 'ZY'), ('IZ', 'ZZ'),
    ('XI', 'XX'), ('XX', 'XI'), ('XY', 'YZ'), ('XZ', 'YI'),
    ('YI', 'YX'), ('YX', 'YI'), ('YY', 'XZ'), ('YZ', 'XI'),
    ('ZI', 'ZI'), ('ZX', 'ZX'), ('ZY', 'IY'), ('ZZ', 'IZ'),
]

def apply_pauli(qc, pauli, qubit):
    """Apply a single Pauli gate (I=nothing, X, Y, Z) to a qubit."""
    if   pauli == 'X': qc.x(qubit)
    elif pauli == 'Y': qc.y(qubit)
    elif pauli == 'Z': qc.z(qubit)
    # 'I' → do nothing

def twirl_circuit(qc, seed=0):
    """
    Return a new circuit with each CX gate twirled by a random Pauli pair.
    All other gates are copied unchanged.
    """
    rng    = np.random.default_rng(seed)
    twirled = QuantumCircuit(qc.num_qubits, qc.num_clbits)

    for instr in qc.data:
        if instr.operation.name == 'cx':
            ctrl = qc.find_bit(instr.qubits[0]).index
            tgt  = qc.find_bit(instr.qubits[1]).index

            # pick a random twirl pair
            pre, post = TWIRL_TABLE[rng.integers(len(TWIRL_TABLE))]

            # insert input Paulis before CX
            apply_pauli(twirled, pre[0],  ctrl)
            apply_pauli(twirled, pre[1],  tgt)

            # the CX gate itself
            twirled.cx(ctrl, tgt)

            # insert correction Paulis after CX
            apply_pauli(twirled, post[0], ctrl)
            apply_pauli(twirled, post[1], tgt)

        else:
            # copy all other gates unchanged
            twirled.append(instr.operation,
                           [qc.find_bit(q).index for q in instr.qubits],
                           [qc.find_bit(c).index for c in instr.clbits])

    return twirled

# ── 1. Build the original circuit ────────────────────────────────────────────
qc = QuantumCircuit(3, name="Original")
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.h(2)

print("=" * 55)
print("ORIGINAL CIRCUIT")
print("=" * 55)
print(qc.draw())
print(f"Depth: {qc.depth()}   Gates: {dict(qc.count_ops())}")
print()

# ── 2. Show three twirlings ───────────────────────────────────────────────────
twirled_circuits = {}
for seed in [0, 1, 2]:
    t = twirl_circuit(qc, seed=seed)
    t.name = f"Twirled seed={seed}"
    twirled_circuits[seed] = t
    print(f"TWIRLED (seed={seed})")
    print(t.draw())
    print(f"Depth: {t.depth()}   Gates: {dict(t.count_ops())}")
    print()

# ── 3. Print the twirl choice for each CX ────────────────────────────────────
print("=" * 55)
print("Twirl choices per CX gate (seed=0)")
print("=" * 55)
rng = np.random.default_rng(0)
cx_count = sum(1 for i in qc.data if i.operation.name == 'cx')
for i in range(cx_count):
    pre, post = TWIRL_TABLE[rng.integers(len(TWIRL_TABLE))]
    ctrl_pauli, tgt_pauli = pre[0], pre[1]
    ctrl_corr,  tgt_corr  = post[0], post[1]
    print(f"  CX {i+1}: insert ({ctrl_pauli},{tgt_pauli}) before"
          f" → correction ({ctrl_corr},{tgt_corr}) after")
print()

# ── 4. Visual comparison ──────────────────────────────────────────────────────
circuits_to_plot = [qc] + [twirled_circuits[s] for s in [0, 1, 2]]
titles = [
    f"ORIGINAL  |  depth={qc.depth()}  gates={qc.size()}",
    f"TWIRLED seed=0  |  depth={twirled_circuits[0].depth()}  "
    f"gates={twirled_circuits[0].size()}",
    f"TWIRLED seed=1  |  depth={twirled_circuits[1].depth()}  "
    f"gates={twirled_circuits[1].size()}",
    f"TWIRLED seed=2  |  depth={twirled_circuits[2].depth()}  "
    f"gates={twirled_circuits[2].size()}",
]
colors = ["steelblue", "darkorange", "green", "purple"]
tmp_files = []

for i, (circ, title, color) in enumerate(zip(circuits_to_plot, titles, colors)):
    f = circ.draw(output="mpl", style="clifford", fold=-1)
    fname = f"_tmp_twirl_{i}.png"
    f.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close(f)
    tmp_files.append((fname, title, color))

fig, axes = plt.subplots(4, 1, figsize=(14, 16))
fig.suptitle(
    "Pauli Twirling — same circuit, different random Pauli wrappings\n"
    "(CX gates unchanged logically — only noise is symmetrised)",
    fontsize=12, fontweight="bold"
)
for ax, (fname, title, color) in zip(axes, tmp_files):
    img = plt.imread(fname)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=9, fontweight="bold", color=color, pad=4)

plt.tight_layout()
plt.savefig("pauli_twirl_demo.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: pauli_twirl_demo.png")

# ── 5. Cleanup ────────────────────────────────────────────────────────────────
import os
for fname, _, _ in tmp_files:
    if os.path.exists(fname):
        os.remove(fname)
