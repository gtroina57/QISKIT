"""Execute the SSW GAN notebook while preserving previous experiment notebooks."""
import hashlib
import argparse
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--notebook',default='GAN_03_SSW_quantum_generator.ipynb')
args=parser.parse_args()
originals=['GAN_01.ipynb','GAN_01_latent4.ipynb','GAN_01_latent4_monitored.ipynb','GAN_02_BiGAN.ipynb']
if args.notebook != 'GAN_03_SSW_quantum_generator.ipynb':
    originals.append('GAN_03_SSW_quantum_generator.ipynb')
if args.notebook not in ['GAN_03_SSW_quantum_generator.ipynb','GAN_04_SSW_quantum_generator_longer.ipynb']:
    originals.append('GAN_04_SSW_quantum_generator_longer.ipynb')
hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in originals}
path=root/args.notebook
n=nbformat.read(path,as_version=4)
out=root/'ssw_gan_results'
out.mkdir(exist_ok=True)
manager=KernelManager(kernel_name='python3')
manager._kernel_spec=KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                               display_name='DOUBT_QC',language='python',env={'IPYTHONDIR':str(out/'ipython')})
client=NotebookClient(n,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(root)}})
client.km=manager
print(f'Executing {args.notebook}: validation followed by configured GAN training.',flush=True)
try:
    client.execute()
finally:
    if manager.has_kernel: manager.shutdown_kernel(now=True)
    nbformat.write(n,path)
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
print('SSW notebook completed; previous notebooks unchanged.',flush=True)
