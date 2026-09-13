from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
import matplotlib.pyplot as plt


# ── 1. Build the abstract Bell state circuit ──────────────────────────────────
qc = QuantumCircuit(2, name="Original")
qc.h(0)
qc.cx(0, 1)

# ── 2. Transpile at all optimization levels ───────────────────────────────────
basis_gates = ["ecr", "rz", "sx", "x"]
transpiled  = {}

for level in [0, 1, 2, 3]:
    pm  = generate_preset_pass_manager(
        optimization_level=level,
        basis_gates=basis_gates,
        seed_transpiler=42,
    )
    tqc = pm.run(qc)
    tqc.name = f"Level {level}"
    transpiled[level] = tqc

# ── 3. Print stats table ──────────────────────────────────────────────────────
print("=" * 65)
print(f"{'Metric':<20} {'Original':>10}  {'L0':>6}  {'L1':>6}  {'L2':>6}  {'L3':>6}")
print("=" * 65)
metrics = {
    "Depth":      lambda c: c.depth(),
    "Gate count": lambda c: c.size(),
    "ECR count":  lambda c: c.count_ops().get("ecr", 0),
    "Rz count":   lambda c: c.count_ops().get("rz",  0),
    "SX count":   lambda c: c.count_ops().get("sx",  0),
}
for name, fn in metrics.items():
    vals = "  ".join(f"{fn(transpiled[l]):>6}" for l in range(4))
    print(f"{name:<20} {fn(qc):>10}  {vals}")
print("=" * 65)

print("\nGate breakdown per level:")
for level in range(4):
    ops = transpiled[level].count_ops()
    print(f"  Level {level}: {dict(ops)}")

# ── 4. Save each circuit as its own PNG ───────────────────────────────────────
image_files = []

fig = qc.draw(output="mpl", fold=-1, style="clifford")
fig.suptitle("ORIGINAL CIRCUIT  —  abstract (H + CX)",
             fontsize=11, fontweight="bold", color="steelblue")
fig.savefig("circ_original.png", dpi=120, bbox_inches="tight")
plt.close(fig)
image_files.append(("circ_original.png", "steelblue",
    f"ORIGINAL  |  depth={qc.depth()}  gates={qc.size()}"))

colors = ["#c0392b", "#e67e22", "#27ae60", "#8e44ad"]
descs  = [
    "LEVEL 0  —  no optimization: direct decomposition to ECR basis",
    "LEVEL 1  —  light: cancels redundant single-qubit gates",
    "LEVEL 2  —  medium: better layout, more cancellation",
    "LEVEL 3  —  heavy: maximum optimization, fewest gates",
]

for level in range(4):
    tqc   = transpiled[level]
    depth = tqc.depth()
    gates = tqc.size()
    ecr   = tqc.count_ops().get("ecr", 0)
    fname = f"circ_level{level}.png"

    fig = tqc.draw(output="mpl", fold=-1, style="clifford", idle_wires=False)
    fig.suptitle(
        f"{descs[level]}\ndepth={depth}  gates={gates}  ECR={ecr}",
        fontsize=9, fontweight="bold", color=colors[level]
    )
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    image_files.append((fname, colors[level],
        f"Level {level}  |  depth={depth}  gates={gates}  ECR={ecr}"))

# ── 5. Summary figure ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(14, 18))
fig.suptitle(
    "Bell State Transpilation — H+CX → ECR basis [ecr, rz, sx, x]",
    fontsize=13, fontweight="bold", y=0.99
)

for ax, (fname, color, title) in zip(axes, image_files):
    img = plt.imread(fname)
    ax.imshow(img)
    ax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=6)
    ax.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("transpiling_demo.png", dpi=120, bbox_inches="tight")
plt.show()

# ── 6. Bar chart ──────────────────────────────────────────────────────────────
import numpy as np

levels      = ["Original", "Level 0", "Level 1", "Level 2", "Level 3"]
circuits    = [qc] + [transpiled[l] for l in range(4)]
depths      = [c.depth() for c in circuits]
gate_counts = [c.size()  for c in circuits]
ecr_counts  = [c.count_ops().get("ecr", 0) for c in circuits]
rz_counts   = [c.count_ops().get("rz",  0) for c in circuits]
sx_counts   = [c.count_ops().get("sx",  0) for c in circuits]

x = np.arange(len(levels))
w = 0.15
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.bar(x - 2*w, depths,      w, label="Depth",      color="steelblue")
ax2.bar(x - 1*w, gate_counts, w, label="Gate count", color="orange")
ax2.bar(x + 0*w, ecr_counts,  w, label="ECR count",  color="green")
ax2.bar(x + 1*w, rz_counts,   w, label="Rz count",   color="red")
ax2.bar(x + 2*w, sx_counts,   w, label="SX count",   color="purple")
ax2.set_xticks(x)
ax2.set_xticklabels(levels)
ax2.set_ylabel("Count")
ax2.set_title("Bell State — Circuit Metrics vs Optimization Level",
              fontweight="bold")
ax2.legend()
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("transpiling_metrics.png", dpi=120, bbox_inches="tight")
plt.show()

# ── 7. Cleanup temp files ─────────────────────────────────────────────────────
import os
for fname, _, _ in image_files:
    if os.path.exists(fname):
        os.remove(fname)
