from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler import generate_preset_pass_manager

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)

# Create a dynamic circuit

qubits = QuantumRegister(1)
clbits = ClassicalRegister(1)
qc = QuantumCircuit(qubits, clbits)
(q0,) = qubits
(c0,) = clbits

qc.h(q0)
qc.measure(q0, c0)
with qc.if_test((c0, 1)):
    qc.x(q0)
qc.measure(q0, c0)

# Convert to an ISA circuit for the given backend

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)

# Generate samplers for backend targets
sampler = SamplerV2(backend)
sampler.options.experimental = {"execution": {"scheduler_timing": True}}

# Submit jobs
sampler_job = sampler.run([isa_circuit])
result = sampler_job.result()

print(
    f">>> {' Job ID:':<10}  {sampler_job.job_id()} ({sampler_job.status()})"
)

# Get the circuit schedule timing
result[0].metadata["compilation"]["scheduler_timing"]["timing"]

from qiskit_ibm_runtime.visualization import draw_circuit_schedule_timing

circuit_schedule = result[0].metadata["compilation"]["scheduler_timing"][
    "timing"
]
fig = draw_circuit_schedule_timing(
    circuit_schedule=circuit_schedule,
    included_channels=None,
    filter_readout_channels=False,
    filter_barriers=False,
    width=1000,
)

# Uncomment the following line to display the figure
# 
# fig.show(renderer="notebook")

# Save to a file
# fig.write_html("scheduler_timing.html")