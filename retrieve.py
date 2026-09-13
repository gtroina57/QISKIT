import datetime
from qiskit_ibm_runtime import QiskitRuntimeService

three_months_ago = datetime.datetime.now() - datetime.timedelta(days=90)

service = QiskitRuntimeService()
jobs_in_last_three_months = service.jobs(created_after=three_months_ago)

print(f"Total jobs found: {len(jobs_in_last_three_months)}")
print()

for job in jobs_in_last_three_months[:15]:
    print(f"Job ID  : {job.job_id()}")
    print(f"Status  : {job.status()}")
    print(f"Backend : {job.backend().name}")
    print(f"Created : {job.creation_date}")
    print("-" * 40)

# Get ID of most recent successful job for demonstration.
# This will not work if you've never successfully run a job.
successful_job = next(
    j for j in service.jobs(limit=1000) if j.status() == "DONE"
)
job_id = successful_job.job_id()
print(job_id)

retrieved_job = service.job(job_id)
retrieved_job.result()

import json
from qiskit_ibm_runtime import RuntimeEncoder

with open("result.json", "w") as file:
    json.dump(retrieved_job.result(), file, cls=RuntimeEncoder)