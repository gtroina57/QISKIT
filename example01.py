from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit_ibm_runtime.options import EstimatorOptions
import matplotlib.pyplot as plt
import numpy as np
 
qc = QuantumCircuit(2) 
qc.h(0) 
qc.cx(0,1) 
pass_manager = generate_preset_pass_manager( 
optimization_level=3, 
coupling_map=[[0, 1], [1, 2]] , 
basis_gates=['h', 'swap', 'cx'], 
initial_layout=[0, 2] 
) 
tqc = pass_manager.run(qc) 
tqc.draw(output="mpl")  

plt.show()
