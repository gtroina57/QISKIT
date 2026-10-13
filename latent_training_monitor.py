"""Deterministic diagnostics; no changes to qGAN losses or optimizers."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from compare_qgan_convergence import evaluate


class LatentTrainingMonitor:
    def __init__(self, scope, output_dir, evaluation_count=64):
        before_rng = torch.get_rng_state().clone()
        self.scope = scope
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        rng = torch.Generator().manual_seed(98765)
        self.fixed_z = (2 * torch.rand(evaluation_count, 4, generator=rng) - 1).numpy() * np.pi
        self.history = []
        self.best_kl = float("inf")
        self.best_epoch = None
        assert torch.equal(before_rng, torch.get_rng_state())

    def checkpoint(self, epoch, metrics, probabilities):
        return {
            "epoch": epoch, "generator": copy.deepcopy(self.scope["generator"].state_dict()),
            "discriminator": copy.deepcopy(self.scope["discriminator"].state_dict()),
            "generator_optimizer": copy.deepcopy(self.scope["generator_optimizer"].state_dict()),
            "discriminator_optimizer": copy.deepcopy(self.scope["discriminator_optimizer"].state_dict()),
            "torch_rng_state": torch.get_rng_state().clone(),
            "fixed_z": torch.from_numpy(self.fixed_z.copy()),
            "metrics": metrics, "exact_marginal": torch.from_numpy(probabilities.copy()),
            "sampler_seed": self.scope["sampler"].seed,
            "replay_note": "Unseeded sampler draws cannot be reproduced exactly from this checkpoint.",
        }

    def record(self, epoch, training_batch_kl=None):
        before = torch.get_rng_state().clone()
        weights = self.scope["generator"].weight.detach().clone()
        metrics, probabilities = evaluate(self.scope, True, self.fixed_z)
        assert torch.equal(before, torch.get_rng_state())
        assert torch.equal(weights, self.scope["generator"].weight)
        entry = {"epoch": epoch, "training_batch_kl": training_batch_kl, **metrics}
        self.history.append(entry)
        state = self.checkpoint(epoch, metrics, probabilities)
        # Best model is chosen using the exact uniform-prior marginal, not a noisy training batch.
        if metrics["kl_generated_to_target"] < self.best_kl:
            self.best_kl = metrics["kl_generated_to_target"]
            self.best_epoch = epoch
            torch.save(state, self.out / "best.pt")
        torch.save(state, self.out / "latest.pt")
        (self.out / "history.json").write_text(json.dumps(self.history, indent=2) + "\n", encoding="utf-8")
        print(f"Epoch {epoch}: training batch KL={training_batch_kl}; "
              f"fixed KL={metrics['fixed_latent_kl']:.5f}; "
              f"exact marginal KL={metrics['kl_generated_to_target']:.5f}; "
              f"best epoch={self.best_epoch}", flush=True)
        return entry

    def finish(self):
        final = torch.load(self.out / "latest.pt", weights_only=True)
        torch.save(final, self.out / "final.pt")
        report = {
            "best_epoch": self.best_epoch, "best_exact_kl": self.best_kl,
            "final_epoch": final["epoch"], "final_metrics": final["metrics"],
            "best_to_final_kl_increase": final["metrics"]["kl_generated_to_target"] - self.best_kl,
            "fixed_evaluation_vectors": len(self.fixed_z),
            "selection_metric": "KL of noiseless exact uniform-prior marginal",
            "training_unchanged": True, "sampler_seed": self.scope["sampler"].seed,
        }
        (self.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        e = np.array([r["epoch"] for r in self.history])
        noisy = [r["training_batch_kl"] for r in self.history]
        axes[0].plot(e[1:], noisy[1:], alpha=0.45, label="Fresh training batch (before update)")
        axes[0].plot(e, [r["fixed_latent_kl"] for r in self.history], label="Fixed 64-vector evaluation")
        axes[0].plot(e, [r["kl_generated_to_target"] for r in self.history], label="Exact prior marginal")
        axes[0].axvline(self.best_epoch, linestyle="--", color="gray", label="Best checkpoint")
        axes[0].set_ylabel("KL(generated || target)")
        axes[0].legend(fontsize=7)
        axes[1].plot(e, [r["total_variation"] for r in self.history])
        axes[1].set_ylabel("Exact marginal total variation")
        for ax in axes:
            ax.set_xlabel("Completed training epochs")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
        fig.savefig(self.out / "monitoring.png", dpi=150, bbox_inches="tight")
        if plt.get_backend().lower() == "agg":
            plt.close(fig)
        else:
            plt.show()
        return report
