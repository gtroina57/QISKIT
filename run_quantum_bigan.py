"""Run a reproducible BiGAN experiment; defaults to a short development run."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import time

import torch
from quantum_bigan import QuantumBiGAN, evaluate, make_images_and_labels, train, verify_dataset_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=Path("bigan_results"))
    args = parser.parse_args()
    if not (1 <= args.train_count <= 256 and args.epochs > 0 and args.batch_size > 0
            and args.lr > 0 and args.shots > 0):
        parser.error("Require 1..256 training examples and positive epochs, batch size, lr, shots")
    torch.set_num_threads(2)
    start = time.perf_counter()
    original = Path(__file__).resolve().parent / "GAN_01.ipynb"
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    dataset = verify_dataset_source()
    train_images, train_labels = make_images_and_labels(256, 42)
    test_images, test_labels = make_images_and_labels(128, 123)
    # Fixed subset of the training split; never select using held-out metrics.
    order = torch.randperm(256, generator=torch.Generator().manual_seed(31415))
    selection = order[:args.train_count]
    model = QuantumBiGAN(args.seed, args.shots)
    print("Evaluating untrained baseline on original held-out split", flush=True)
    initial, _ = evaluate(model, train_images, train_labels, test_images, test_labels)
    history = train(model, train_images[selection], args.epochs, args.batch_size, args.lr)
    print("Evaluating trained model on all 128 held-out images", flush=True)
    final, examples = evaluate(model, train_images, train_labels, test_images, test_labels)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = {
        "status": "completed", "run_type": "development run; convergence not established",
        "config": config, "dataset": dataset, "gan_training_indices": selection.tolist(),
        "probe_train_count": 256, "test_count": 128,
        "initial": initial, "final": final, "history": history,
        "original_notebook_sha256": digest,
        "versions": {p: version(p) for p in ("qiskit", "qiskit-machine-learning", "torch", "numpy")},
        "elapsed_seconds": time.perf_counter() - start,
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({"model": model.state_dict(), "config": config, "dataset": dataset,
                "history": history, "gan_training_indices": selection,
                "initial": initial, "final": final}, args.output_dir / "quantum_bigan.pt")
    torch.save(examples, args.output_dir / "examples.pt")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 8, figsize=(12, 5), constrained_layout=True)
    for row, name in enumerate(("real", "generated", "reconstructed")):
        for col, image in enumerate(examples[name]):
            axes[row, col].imshow(image[0], vmin=0, vmax=1.6, cmap="gray")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(name)
    fig.suptitle("Quantum BiGAN development run — shared intensity scale [0, 1.6]")
    fig.savefig(args.output_dir / "examples.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"initial": initial, "final": final,
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)
    print(f"Saved results in {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
