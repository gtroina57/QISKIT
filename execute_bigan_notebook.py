"""Execute and save the separate BiGAN notebook, preserving the originals."""
from pathlib import Path
import hashlib
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
originals = ("GAN_01.ipynb", "GAN_01_latent4.ipynb", "I_JEPA_05.ipynb")
hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in originals}
path = root / "GAN_02_BiGAN.ipynb"
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=1800, kernel_name="python3",
                        resources={"metadata": {"path": str(root)}})
# Pin the kernel to the interpreter used to launch this validation.
import sys
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec
manager = KernelManager(kernel_name="python3")
manager._kernel_spec = KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                  display_name="DOUBT_QC", language="python")
client.km = manager
try:
    client.execute()
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
nbformat.validate(notebook)
assert all(hashlib.sha256((root / p).read_bytes()).hexdigest() == hashes[p] for p in originals)
nbformat.write(notebook, path)
print("BiGAN notebook executed, validated, and saved; original notebooks unchanged.", flush=True)
