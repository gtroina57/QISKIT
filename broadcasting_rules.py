import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

def draw_grid(ax, rows, cols, color, label, alpha=0.8):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(mpatches.FancyBboxPatch(
                (c + 0.05, r + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="white",
                linewidth=1.5, alpha=alpha
            ))
    ax.set_xlim(0, max(cols, 1))
    ax.set_ylim(0, max(rows, 1))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(label, fontsize=10, fontweight="bold")

def draw_arrow(fig, ax_left, ax_right, label):
    # Draw "+" and "=" as text between axes
    pass

fig = plt.figure(figsize=(16, 14))
fig.suptitle("Qiskit Primitives — Broadcasting Rules", fontsize=15, fontweight="bold", y=0.98)

gs = GridSpec(4, 7, figure=fig, hspace=0.6, wspace=0.3)

# Colors
COL_A   = "#4C9BE8"   # blue  — A array
COL_B   = "#E87C4C"   # orange — B array
COL_RES = "#6DBF6D"   # green — result
COL_STR = "#B0C4DE"   # light blue — stretched dimension

def add_label(ax, text, fontsize=14):
    ax.text(0.5, 0.5, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", transform=ax.transAxes)
    ax.axis("off")

# ── Rule 1: (4,1) + (1,21) → (4,21) — size-1 stretches ──────────────────────
ax1a = fig.add_subplot(gs[0, 0])
ax1b = fig.add_subplot(gs[0, 1])
ax1c = fig.add_subplot(gs[0, 2])
ax1d = fig.add_subplot(gs[0, 3])
ax1e = fig.add_subplot(gs[0, 4])
ax1f = fig.add_subplot(gs[0, 5:])

draw_grid(ax1a, 4, 1, COL_A,   "(4, 1)\nObservables")
add_label(ax1b, "+")
draw_grid(ax1c, 1, 5, COL_B,   "(1, 5)*\nParams\n*shown as 5 for clarity")
add_label(ax1d, "→")
draw_grid(ax1e, 4, 5, COL_RES, "(4, 5)\nResult")
ax1f.text(0.0, 0.5,
    "Rule 1 — Size-1 stretches:\n"
    "• (4,1): the col dimension=1 stretches to 5\n"
    "• (1,5): the row dimension=1 stretches to 4\n"
    "• Every observable paired with every param set",
    va="center", fontsize=9, transform=ax1f.transAxes,
    bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
ax1f.axis("off")

# ── Rule 2: (3,5) + (3,5) → (3,5) — equal dimensions match one-to-one ────────
ax2a = fig.add_subplot(gs[1, 0])
ax2b = fig.add_subplot(gs[1, 1])
ax2c = fig.add_subplot(gs[1, 2])
ax2d = fig.add_subplot(gs[1, 3])
ax2e = fig.add_subplot(gs[1, 4])
ax2f = fig.add_subplot(gs[1, 5:])

draw_grid(ax2a, 3, 5, COL_A,   "(3, 5)\nObservables")
add_label(ax2b, "+")
draw_grid(ax2c, 3, 5, COL_B,   "(3, 5)\nParams")
add_label(ax2d, "→")
draw_grid(ax2e, 3, 5, COL_RES, "(3, 5)\nResult")
ax2f.text(0.0, 0.5,
    "Rule 2 — Equal dimensions match as-is:\n"
    "• Both arrays have the same shape (3,5)\n"
    "• Element [i,j] of A pairs with element [i,j] of B\n"
    "• One-to-one pairing, no stretching needed",
    va="center", fontsize=9, transform=ax2f.transAxes,
    bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
ax2f.axis("off")

# ── Rule 3: (3,) + (5,2) → (3,5) — missing dims treated as 1 ─────────────────
ax3a = fig.add_subplot(gs[2, 0])
ax3b = fig.add_subplot(gs[2, 1])
ax3c = fig.add_subplot(gs[2, 2])
ax3d = fig.add_subplot(gs[2, 3])
ax3e = fig.add_subplot(gs[2, 4])
ax3f = fig.add_subplot(gs[2, 5:])

draw_grid(ax3a, 3, 1, COL_A,   "(3,)\nObservables\n(treated as (3,1))")
add_label(ax3b, "+")
draw_grid(ax3c, 1, 5, COL_B,   "(5, 2)*\nParams\n*shown as (1,5)")
add_label(ax3d, "→")
draw_grid(ax3e, 3, 5, COL_RES, "(3, 5)\nResult")
ax3f.text(0.0, 0.5,
    "Rule 3 — Missing dims treated as 1:\n"
    "• Shape (3,) is treated as (3,1)\n"
    "• Then Rule 1 applies: (3,1) + (1,5) → (3,5)\n"
    "• Qiskit automatically promotes lower-dim arrays",
    va="center", fontsize=9, transform=ax3f.transAxes,
    bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
ax3f.axis("off")

# ── Rule 4: incompatible → ERROR ───────────────────────────────────────────────
ax4a = fig.add_subplot(gs[3, 0])
ax4b = fig.add_subplot(gs[3, 1])
ax4c = fig.add_subplot(gs[3, 2])
ax4d = fig.add_subplot(gs[3, 3])
ax4e = fig.add_subplot(gs[3, 4])
ax4f = fig.add_subplot(gs[3, 5:])

draw_grid(ax4a, 3, 5, COL_A,   "(3, 5)\nObservables")
add_label(ax4b, "+")
draw_grid(ax4c, 4, 5, COL_B,   "(4, 5)\nParams")
add_label(ax4d, "→")

# Error box
ax4e.add_patch(mpatches.FancyBboxPatch(
    (0.1, 0.2), 0.8, 0.6,
    boxstyle="round,pad=0.05",
    facecolor="#FF6B6B", edgecolor="darkred", linewidth=2
))
ax4e.text(0.5, 0.5, "ERROR", ha="center", va="center",
          fontsize=11, fontweight="bold", color="white",
          transform=ax4e.transAxes)
ax4e.set_title("Incompatible\nshapes", fontsize=10, fontweight="bold")
ax4e.axis("off")

ax4f.text(0.0, 0.5,
    "Rule 4 — Incompatible dimensions → ERROR:\n"
    "• Row dims are 3 and 4: neither is 1\n"
    "• Cannot stretch — shapes are incompatible\n"
    "• Fix: reshape one array so a dim becomes 1",
    va="center", fontsize=9, transform=ax4f.transAxes,
    bbox=dict(boxstyle="round", facecolor="#ffe0e0", alpha=0.8))
ax4f.axis("off")

# Row labels on the left
for i, label in enumerate(["Rule 1\nStretch", "Rule 2\nMatch", "Rule 3\nPromote", "Rule 4\nError"]):
    ax = fig.add_subplot(gs[i, 6])
    ax.axis("off")

plt.savefig("broadcasting_rules.png", dpi=150, bbox_inches="tight")
plt.show()
