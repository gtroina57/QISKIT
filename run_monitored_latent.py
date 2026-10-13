"""Smoke-test or fully run the actual monitored notebook's model/training cells."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch

os.environ.setdefault("MPLBACKEND", "Agg")

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
          for p in ("GAN_01.ipynb", "GAN_01_latent4.ipynb")}
n = json.loads((root / "GAN_01_latent4_monitored.ipynb").read_text(encoding="utf-8"))
sources = ["".join(c["source"]) for c in n["cells"] if c["cell_type"] == "code"]
scope = {}
torch.set_num_threads(2)
for index in (0, 1, 2, 4, 7, 8, 9, 10, 11, 12, 13):
    source = sources[index]
    if args.smoke and index == 13:
        source = source.replace('Path("latent_monitor_results")', 'Path("latent_monitor_smoke")')
    exec(compile(source, f"monitored-cell-{index}", "exec"), scope)
scope["plot_training_progress"] = lambda: None
source = sources[15].replace("n_epochs = 50", f"n_epochs = {args.epochs}")
exec(compile(source, "monitored-training", "exec"), scope)
monitor = scope["monitor"]
state = torch.load(monitor.out / "best.pt", weights_only=True)
old = scope["generator"].weight.detach().clone()
scope["generator"].load_state_dict(state["generator"])
from compare_qgan_convergence import evaluate
metrics, _ = evaluate(scope, True, monitor.fixed_z)
assert abs(metrics["kl_generated_to_target"] - monitor.best_kl) < 1e-10
scope["generator"].weight.data.copy_(old)
assert all(hashlib.sha256((root / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
print("Checkpoint reload metric verified; source notebooks unchanged.", flush=True)
