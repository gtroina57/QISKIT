"""Render completed matched experiments and summarize their evidence."""
import json
import hashlib
from importlib.metadata import version
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "convergence_results"


def main():
    paths = sorted(OUT.glob("*.json")) + sorted((ROOT / "convergence_longer").glob("*.json"))
    records = [json.loads(p.read_text(encoding="utf-8")) for p in paths
               if p.name != "summary.json"]
    records = [r for r in records if "final_probabilities" in r]
    if not records:
        raise RuntimeError("No completed convergence runs")
    for seed in sorted({r["seed"] for r in records}):
        paired = [r for r in records if r["seed"] == seed]
        assert all(r["initial_weights"] == paired[0]["initial_weights"] for r in paired)
        assert all(r["initial_discriminator"] == paired[0]["initial_discriminator"] for r in paired)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    table = []
    for r in records:
        e = r["evaluation"]
        variant_label = {"original": "original", "latent4": "4D latent, batch 4", "latent16": "4D latent, batch 16"}[r["variant"]]
        label = f"{variant_label} / seed {r['seed']}"
        for ax, metric in zip(axes, ("kl_generated_to_target", "total_variation")):
            if r["epochs"] == 50:
                ax.plot([a["epoch"] for a in e], [a[metric] for a in e], label=label)
            ax.set_xlabel("Training epoch")
            ax.grid(alpha=0.3)
        final = e[-1]
        table.append({"variant": r["variant"], "seed": r["seed"], "epochs": r["epochs"],
                      "latent_dim": 0 if r["variant"] == "original" else 4,
                      "latent_batch_size": r["latent_batch_size"],
                      "initial_kl": e[0]["kl_generated_to_target"], "final_kl": final["kl_generated_to_target"],
                      "initial_tv": e[0]["total_variation"], "final_tv": final["total_variation"],
                      "best_kl": min(a["kl_generated_to_target"] for a in e),
                      "final_fixed_vs_exact_tv": final["fixed_vs_exact_tv"],
                      "elapsed_seconds": r["elapsed_seconds"]})
    axes[0].set_ylabel("KL(generated || target)")
    axes[1].set_ylabel("Total variation from Gaussian target")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.autoscale(enable=True, axis="y")
        ax.set_ylim(bottom=0)
    fig.suptitle("Gaussian qGAN comparison — noiseless evaluation")
    fig.savefig(OUT / "convergence.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    longer = [r for r in records if r["epochs"] > 50]
    if longer:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for r in longer:
            e = r["evaluation"]
            for ax, metric in zip(axes, ("kl_generated_to_target", "total_variation")):
                ax.plot([a["epoch"] for a in e], [a[metric] for a in e])
                ax.axvline(50, color="gray", linestyle="--", label="Original 50-epoch duration")
                ax.set_xlabel("Training epoch")
                ax.grid(alpha=0.3)
        axes[0].set_ylabel("KL(generated || target)")
        axes[1].set_ylabel("Total variation")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.autoscale(enable=True, axis="y")
            ax.set_ylim(bottom=0)
        fig.suptitle("Longer original-generator control — noiseless evaluation")
        fig.savefig(OUT / "longer_original.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    probabilities = [np.array(records[0]["target_probabilities"])] + [np.array(r["final_probabilities"]) for r in records]
    rows = int(np.ceil(len(probabilities) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(10, 3 * rows), constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    maximum = max(p.max() for p in probabilities)
    for ax, p, label in zip(axes, probabilities, ["Gaussian target"] + [f"{r['variant']}, {r['epochs']} epochs\nseed {r['seed']}" for r in records]):
        edge = 2 + (4 / 7) / 2
        image = ax.imshow(p.reshape(8, 8), origin="lower", extent=(-edge, edge, -edge, edge), vmin=0, vmax=maximum, cmap="viridis")
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    for ax in axes[len(probabilities):]:
        ax.set_visible(False)
    fig.colorbar(image, ax=list(axes[:len(probabilities)]), label="Probability per grid point", shrink=0.7)
    fig.savefig(OUT / "distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    summary = {"matched_initial_parameters": True,
               "versions": {p: version(p) for p in ("qiskit", "qiskit-machine-learning", "torch", "numpy", "scipy")},
               "source_notebook_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                           for p in ("GAN_01.ipynb", "GAN_01_latent4.ipynb")},
               "evaluation": "Noiseless statevector probabilities; exact uniform-prior marginal for latent generator",
               "fixed_latent_evaluation_count": 256,
               "training": "Original notebook optimizer and losses; 1024 sampler shots; independent seeded latent RNG",
               "runs": table}
    diagnostic_paths = sorted((ROOT / "convergence_direct_kl").glob("*.json")) + sorted((ROOT / "convergence_direct_kl_longer").glob("*.json"))
    diagnostics = [json.loads(p.read_text(encoding="utf-8")) for p in diagnostic_paths]
    diagnostics = [r for r in diagnostics if "final_probabilities" in r]
    if diagnostics:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        summary["direct_target_diagnostics"] = []
        for r in diagnostics:
            matched = next(a for a in records if a["variant"] == r["variant"]
                           and a["seed"] == r["seed"] and a["epochs"] == r["epochs"])
            assert r["initial_weights"] == matched["initial_weights"]
            assert r["initial_discriminator"] == matched["initial_discriminator"]
            for ax, metric in zip(axes, ("kl_generated_to_target", "total_variation")):
                for curve, style, name in ((matched, "--", "adversarial"), (r, "-", "direct KL diagnostic")):
                    e = curve["evaluation"]
                    ax.plot([a["epoch"] for a in e], [a[metric] for a in e], linestyle=style,
                            label=f"{r['variant']}, {r['epochs']} epochs: {name}")
                ax.set_xlabel("Training epoch")
                ax.grid(alpha=0.3)
            summary["direct_target_diagnostics"].append({"variant": r["variant"], "epochs": r["epochs"],
                "seed": r["seed"], "final": r["evaluation"][-1],
                "matched_adversarial_final": matched["evaluation"][-1]})
        axes[0].set_ylabel("KL(generated || target)")
        axes[1].set_ylabel("Total variation")
        axes[1].legend(fontsize=6)
        for ax in axes:
            ax.autoscale(enable=True, axis="y")
            ax.set_ylim(bottom=0)
        fig.suptitle("Direct target fitting is a diagnostic, not GAN training")
        fig.savefig(OUT / "direct_target_diagnostic.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
