# Example:  Running a Quantum Circuit on Real Hardware

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

# --- STEP 1: Authentication ---
# if account is already saved, this will load it
service = QiskitRuntimeService()
# replace <instance_name> with your instance name
# replace <token> with your IBM Quantum API token
# Uncomment and use the following line if you need to specify instance and token
# srvice = QiskitRuntimeService(channel="ibm_cloud",instance="<instance_name>",token=<token>)

   
# --- STEP 2: Backend Selection ---
# Choose the least busy available backend to minimize queue time with real hardware only
backend = service.least_busy(operational=True, simulator=False)
print(f"Using backend: {backend.name}")
print(f"Backend properties: {backend.num_qubits} qubits")

# --- STEP 3: Circuit Preparation ---
# Create a simple Bell state circuit
qc = QuantumCircuit(2, 2)
qc.h(0)                   
qc.cx(0, 1) 
qc.measure([0, 1], [0, 1])

# Display the circuit
qc.draw('mpl')

# Transpile the circuit for the specific hardware backend
# This transpiles the circuit to the backend's: Native gate set,  Qubit connectivity and Optimization level (0-3, where 3 is most optimized)
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
transpiled_circuit = pm.run(qc)

# Define the observable we want to measure (ZZ correlation for example)
ZZ = SparsePauliOp.from_list([("ZZ", 1)])

# --- STEP 4: Job Submission ---
# Create an Estimator instance and run the job
estimator = Estimator(mode=backend)

# Map observables to the transpiled circuit layout
observables = [ZZ.apply_layout(transpiled_circuit.layout)]

# Submit the job
job = estimator.run([(transpiled_circuit, observables)])
print(f"\nJob ID: {job.job_id()}")
print("Job submitted successfully! Check the IBM Quantum dashboard for status.")

# --- STEP 5: Result Retrieval ---
result = job.result()
print(f"Expectation value: {result[0].data.evs}")
