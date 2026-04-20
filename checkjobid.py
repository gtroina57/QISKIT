from qiskit_ibm_runtime import QiskitRuntimeService
import matplotlib.pyplot as plt

observables_labels = ["IZ", "IX", "ZI", "XI", "ZZ", "XX"]
service = QiskitRuntimeService()
job = service.job("d7dmt3h5a5qc73dq2m30")
print(job.status())
result = job.result()
pub_result = result[0]
print("Expectation values:", pub_result.data.evs)

# Plot the result
values = pub_result.data.evs
errors = pub_result.data.stds
# plotting graph
plt.plot(observables_labels, values, "-o")
plt.xlabel("Observables")
plt.ylabel("Values")
plt.show()