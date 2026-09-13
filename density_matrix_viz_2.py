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

# ── 1. Build four interesting states ─────────────────────────────────────────

# |0⟩  — pure ground state
qc0 = QuantumCircuit(1)
rho0 = DensityMatrix(qc0)

# |+⟩  — pure superposition
qc_plus = QuantumCircuit(1)
qc_plus.h(0)
rho_plus = DensityMatrix(qc_plus)

# Bell state (|00⟩ + |11⟩)/√2  — entangled
qc_bell = QuantumCircuit(2)
qc_bell.h(0)
qc_bell.cx(0, 1)
rho_bell = DensityMatrix(qc_bell)


# Bell state (|00⟩ + |11⟩)/√2  — entangled
qc_bell1 = QuantumCircuit(2)
qc_bell1.h(0)
qc_bell1.cx(0, 1)
qc_bell1.z(1)
rho_bell1 = DensityMatrix(qc_bell1)

# ── 2. City plot (real + imag bars) ──────────────────────────────────────────
print("Generating city plots...")
for i, (name, rho) in enumerate(states_1q.items()):
    f = plot_state_city(rho, title=name)
    f.savefig(f"city_{i}.png", dpi=100, bbox_inches="tight")
    plt.close(f)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("City plot — density matrix 3D bars", fontsize=13, fontweight="bold")
for i, (name, _) in enumerate(states_1q.items()):
    img = plt.imread(f"city_{i}.png")
    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(name, fontweight="bold")
plt.tight_layout()
plt.savefig("viz_city.png", dpi=120, bbox_inches="tight")
plt.close()

# ── 3. Hinton plot ────────────────────────────────────────────────────────────
print("Generating Hinton plots...")
for i, (name, rho) in enumerate(states_1q.items()):
    f = plot_state_hinton(rho, title=name)
    f.savefig(f"hinton_{i}.png", dpi=100, bbox_inches="tight")
    plt.close(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("plot_state_hinton — square size = matrix element magnitude", fontsize=13, fontweight="bold")
for i, (name, _) in enumerate(states_1q.items()):
    img = plt.imread(f"hinton_{i}.png")
    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(name, fontweight="bold")
plt.tight_layout()
plt.savefig("viz_hinton.png", dpi=120, bbox_inches="tight")
plt.close()

# ── 4. Pauli vector plot ──────────────────────────────────────────────────────
print("Generating Pauli vector plots...")
for i, (name, rho) in enumerate(states_1q.items()):
    f = plot_state_paulivec(rho, title=name)
    f.savefig(f"pauli_{i}.png", dpi=100, bbox_inches="tight")
    plt.close(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("plot_state_paulivec — expectation values of Pauli operators", fontsize=13, fontweight="bold")
for i, (name, _) in enumerate(states_1q.items()):
    img = plt.imread(f"pauli_{i}.png")
    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(name, fontweight="bold")
plt.tight_layout()
plt.savefig("viz_paulivec.png", dpi=120, bbox_inches="tight")
plt.close()

# ── 5. Bloch sphere ───────────────────────────────────────────────────────────
print("Generating Bloch sphere plots...")
states_bloch = {
    "Ground |0⟩":        Statevector.from_label('0'),
    "Superposition |+⟩": Statevector.from_label('+'),
    "Excited |1⟩":       Statevector.from_label('1'),
}
for i, (name, sv) in enumerate(states_bloch.items()):
    f = plot_bloch_multivector(sv, title=name)
    f.savefig(f"bloch_{i}.png", dpi=100, bbox_inches="tight")
    plt.close(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("plot_bloch_multivector — state as point on Bloch sphere", fontsize=13, fontweight="bold")
for i, (name, _) in enumerate(states_bloch.items()):
    img = plt.imread(f"bloch_{i}.png")
    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(name, fontweight="bold")
plt.tight_layout()
plt.savefig("viz_bloch.png", dpi=120, bbox_inches="tight")
plt.close()

# ── 6. Q-sphere ───────────────────────────────────────────────────────────────
print("Generating Q-sphere plots...")
states_qsphere = {
    "Bell state":     rho_bell,
    "Superposition":  DensityMatrix(qc_plus),
}
for i, (name, rho) in enumerate(states_qsphere.items()):
    f = plot_state_qsphere(rho)
    f.suptitle(name, fontsize=11, fontweight="bold")
    f.savefig(f"qsphere_{i}.png", dpi=100, bbox_inches="tight")
    plt.close(f)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("plot_state_qsphere — amplitudes on sphere surface", fontsize=13, fontweight="bold")
for i, (name, _) in enumerate(states_qsphere.items()):
    img = plt.imread(f"qsphere_{i}.png")
    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(name, fontweight="bold")
plt.tight_layout()
plt.savefig("viz_qsphere.png", dpi=120, bbox_inches="tight")
plt.close()

# ── 7. Histogram ──────────────────────────────────────────────────────────────
print("Generating histogram...")
counts_ideal   = {"00": 500, "11": 500}
counts_noisy   = {"00": 430, "11": 390, "01": 95, "10": 85}
f = plot_histogram(
    [counts_ideal, counts_noisy],
    legend=["Ideal Bell", "Noisy Bell"],
    title="plot_histogram — measurement counts comparison",
    figsize=(8, 5),
)
f.savefig("viz_histogram.png", dpi=120, bbox_inches="tight")
plt.close()

# ── 8. Summary table ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Visualization summary")
print("=" * 60)
summary = [
    ("plot_state_city",        "3D bar chart of density matrix elements"),
    ("plot_state_hinton",      "Square grid: size = magnitude of element"),
    ("plot_state_paulivec",    "Bar chart of ⟨X⟩, ⟨Y⟩, ⟨Z⟩ expectation values"),
    ("plot_bloch_multivector", "Point on Bloch sphere (pure states only)"),
    ("plot_state_qsphere",     "Amplitudes mapped onto sphere surface"),
    ("plot_histogram",         "Measurement outcome counts / distributions"),
]
for name, desc in summary:
    print(f"  {name:<28} {desc}")
print("=" * 60)
print("\nAll plots saved as PNG files in the project directory.")

# Open all saved figures at once
saved = ["viz_city.png", "viz_hinton.png", "viz_paulivec.png",
         "viz_bloch.png", "viz_qsphere.png", "viz_histogram.png"]
import os
for fname in saved:
    img = plt.imread(fname)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(fname, fontweight="bold")
    plt.tight_layout()
plt.show()

# ── 9. Cleanup ────────────────────────────────────────────────────────────────
import os
temp = ([f"city_{i}.png"    for i in range(3)] +
        [f"hinton_{i}.png"  for i in range(3)] +
        [f"pauli_{i}.png"   for i in range(3)] +
        [f"bloch_{i}.png"   for i in range(3)] +
        [f"qsphere_{i}.png" for i in range(2)])
for f in temp:
    if os.path.exists(f):
        os.remove(f)
