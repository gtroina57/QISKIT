from qiskit.circuit import (
    Parameter,
    QuantumCircuit,
    ClassicalRegister,
    QuantumRegister,
)
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives.containers import BitArray
from qiskit.primitives import StatevectorEstimator


import numpy as np

import datetime
from qiskit_ibm_runtime import QiskitRuntimeService

three_months_ago = datetime.datetime.now() - datetime.timedelta(days=90)

service = QiskitRuntimeService()
jobs_in_last_three_months = service.jobs(created_after=three_months_ago)
jobs_in_last_three_months[:3]  # show first three jobs
for job in jobs_in_last_three_months:
    print(job.job_id(), job.status(), job.creation_date)
##plt.show()