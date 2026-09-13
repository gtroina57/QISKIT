from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.visualization import (
    plot_state_city,
    plot_state_hinton,
    plot_state_paulivec,
    plot_state_qsphere,
    plot_bloch_multivector,
    plot_histogram,
)
import matplotlib.pyplot as plt
import os

# ── 1. Build the four Bell states ────────────────────────────────────────────
#
#  |Φ+⟩ = (|00⟩ + |11⟩)/√2   H + CX
#  |Φ-⟩ = (|00⟩ - |11⟩)/√2   H + CX + Z on q0
#  |Ψ+⟩ = (|01⟩ + |10⟩)/√2   H + CX + X on q0
#  |Ψ-⟩ = (|01⟩ - |10⟩)/√2   H + CX + X + Z on q0

def bell_circuit(x_flip=False, z_flip=False):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    if x_flip:
        qc.x(0)
    if z_flip:
        qc.z(0)
    return qc

bell_states = {
    "|Φ+⟩  (|00⟩+|11⟩)/√2": DensityMatrix(bell_circuit()),
    "|Φ-⟩  (|00⟩-|11⟩)/√2": DensityMatrix(bell_circuit(z_flip=True)),
    "|Ψ+⟩  (|01⟩+|10⟩)/√2": DensityMatrix(bell_circuit(x_flip=True)),
    "|Ψ-⟩  (|01⟩-|10⟩)/√2": DensityMatrix(bell_circuit(x_flip=True, z_flip=True)),
}
names  = list(bell_states.keys())
states = list(bell_states.values())

# ── helper: assemble PNGs into one row ───────────────────────────────────────
def assemble_row(fnames, title, out, figsize=(20, 6)):
    fig, axes = plt.subplots(1, len(fnames), figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for ax, (fname, label) in zip(axes, fnames):
        img = plt.imread(fname)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()

# ── 2. City plot ──────────────────────────────────────────────────────────────
print("Generating city plots...")
tmp = []
for i, (name, rho) in enumerate(bell_states.items()):
    f = plot_state_city(rho, title=name)
    fname = f"_tmp_city_{i}.png"
    f.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close(f)
    tmp.append(fname)
assemble_row(list(zip(tmp, names)),
             "plot_state_city — density matrix as 3D bar chart",
             "viz_city.png")

# ── 3. Hinton plot ────────────────────────────────────────────────────────────
print("Generating Hinton plots...")
tmp = []
for i, (name, rho) in enumerate(bell_states.items()):
    f = plot_state_hinton(rho, title=name)
    fname = f"_tmp_hinton_{i}.png"
    f.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close(f)
    tmp.append(fname)
assemble_row(list(zip(tmp, names)),
             "plot_state_hinton — square size = matrix element magnitude",
             "viz_hinton.png")

# ── 4. Pauli vector ───────────────────────────────────────────────────────────
print("Generating Pauli vector plots...")
tmp = []
for i, (name, rho) in enumerate(bell_states.items()):
    f = plot_state_paulivec(rho, title=name)
    fname = f"_tmp_pauli_{i}.png"
    f.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close(f)
    tmp.append(fname)
assemble_row(list(zip(tmp, names)),
             "plot_state_paulivec — Pauli operator expectation values",
             "viz_paulivec.png")

# ── 5. Q-sphere ───────────────────────────────────────────────────────────────
print("Generating Q-sphere plots...")
tmp = []
for i, (name, rho) in enumerate(bell_states.items()):
    f = plot_state_qsphere(rho)
    f.suptitle(name, fontsize=9, fontweight="bold")
    fname = f"_tmp_qsphere_{i}.png"
    f.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close(f)
    tmp.append(fname)
assemble_row(list(zip(tmp, names)),
             "plot_state_qsphere — amplitudes on sphere surface",
             "viz_qsphere.png", figsize=(20, 7))

# ── 6. Bloch multivector (mixed = maximally mixed for Bell — shows origin) ───
print("Generating Bloch sphere plots...")
tmp = []
for i, (name, rho) in enumerate(bell_states.items()):
    f = plot_bloch_multivector(rho, title=name)
    fname = f"_tmp_bloch_{i}.png"
    f.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close(f)
    tmp.append(fname)
assemble_row(list(zip(tmp, names)),
             "plot_bloch_multivector — each qubit's Bloch vector\n"
             "(Bell states are maximally entangled → both qubits at origin)",
             "viz_bloch.png", figsize=(20, 7))

# ── 7. Histogram — simulated measurement counts ───────────────────────────────
print("Generating histograms...")
counts = {
    "|Φ+⟩": {"00": 512, "11": 488},
    "|Φ-⟩": {"00": 495, "11": 505},
    "|Ψ+⟩": {"01": 503, "10": 497},
    "|Ψ-⟩": {"01": 490, "10": 510},
}
f = plot_histogram(
    list(counts.values()),
    legend=list(counts.keys()),
    title="Measurement counts — all four Bell states\n(only correlated pairs appear)",
    figsize=(10, 5),
)
f.savefig("viz_histogram.png", dpi=120, bbox_inches="tight")
plt.close(f)

# ── 8. Summary table ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Visualization summary — Four Bell States")
print("=" * 60)
summary = [
    ("plot_state_city",        "3D bar chart of ρ elements  → viz_city.png"),
    ("plot_state_hinton",      "Square grid of ρ magnitudes → viz_hinton.png"),
    ("plot_state_paulivec",    "⟨XI⟩,⟨YI⟩,⟨ZI⟩... bars    → viz_paulivec.png"),
    ("plot_state_qsphere",     "Amplitudes on sphere        → viz_qsphere.png"),
    ("plot_bloch_multivector", "Bloch vectors per qubit     → viz_bloch.png"),
    ("plot_histogram",         "Measurement counts          → viz_histogram.png"),
]
for name, desc in summary:
    print(f"  {name:<28} {desc}")
print("=" * 60)
print("\nAll plots saved. Opening windows...")

# ── 9. Show all saved figures ─────────────────────────────────────────────────
saved = ["viz_city.png", "viz_hinton.png", "viz_paulivec.png",
         "viz_qsphere.png", "viz_bloch.png", "viz_histogram.png"]
for fname in saved:
    img = plt.imread(fname)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(fname, fontweight="bold")
    plt.tight_layout()
plt.show()

# ── 10. Cleanup temp files ────────────────────────────────────────────────────
temp = ([f"_tmp_city_{i}.png"    for i in range(4)] +
        [f"_tmp_hinton_{i}.png"  for i in range(4)] +
        [f"_tmp_pauli_{i}.png"   for i in range(4)] +
        [f"_tmp_qsphere_{i}.png" for i in range(4)] +
        [f"_tmp_bloch_{i}.png"   for i in range(4)])
for f in temp:
    if os.path.exists(f):
        os.remove(f)
