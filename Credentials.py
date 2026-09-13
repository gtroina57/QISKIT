import getpass

from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    token=getpass.getpass("IBM Quantum token: "),
    instance="open-instance",
    channel="ibm_quantum_platform",
    overwrite=True,
)