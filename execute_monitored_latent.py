"""Execute the separate monitored notebook and preserve the source notebooks."""
import hashlib
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

root = Path(__file__).resolve().parent
sources = ("GAN_01.ipynb", "GAN_01_latent4.ipynb")
hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources}
path = root / "GAN_01_latent4_monitored.ipynb"
(root / "latent_monitor_results").mkdir(parents=True, exist_ok=True)
n = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager._kernel_spec = KernelSpec(
    argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    display_name="DOUBT_QC", language="python",
    env={"IPYTHONDIR": str(root / "latent_monitor_results" / "ipython")},
)
client = NotebookClient(n, timeout=1800, kernel_name="python3",
                        resources={"metadata": {"path": str(root)}})
client.km = manager
print("Executing monitored notebook: unchanged 50-epoch training plus diagnostics.", flush=True)
try:
    client.execute()
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
    nbformat.write(n, path)
assert all(hashlib.sha256((root / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
print("Notebook executed and saved; both source notebooks unchanged.", flush=True)
