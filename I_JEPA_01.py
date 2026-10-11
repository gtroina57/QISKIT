
import numpy as np
import torch

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

torch.manual_seed(42)

# 1. Circuit parameters
n_qubits = 4
x = ParameterVector("x", n_qubits)
theta = ParameterVector("theta", 2 * n_qubits)

qc = QuantumCircuit(n_qubits)

# 2. Encode the input vector
for i in range(n_qubits):
    qc.ry(x[i], i)

# 3. First trainable layer
for i in range(n_qubits):
    qc.ry(theta[i], i)

# 4. Entangling ring
for i in range(n_qubits):
    qc.cz(i, (i + 1) % n_qubits)

# 5. Second trainable layer
for i in range(n_qubits):
    qc.ry(theta[n_qubits + i], i)

print("Quantum circuit:")
print(qc.draw("text"))

# 6. Four independent Z observables
observables = []
for i in range(n_qubits):
    label = "I" * (n_qubits - i - 1) + "Z" + "I" * i
    observables.append(SparsePauliOp(label))

# 7. Create the quantum neural network
qnn = EstimatorQNN(
    circuit=qc,
    estimator=StatevectorEstimator(),
    observables=observables,
    input_params=list(x),
    weight_params=list(theta),
    input_gradients=True
)

# 8. Wrap with PyTorch
model = TorchConnector(
    qnn,
    initial_weights=np.full(8, 0.25)
)

# 9. Input latent vector
z = torch.tensor(
    [[0.2, 0.4, 0.6, 0.8]],
    dtype=torch.float32,
    requires_grad=True
)

# 10. Generate a reachable target using
# the same circuit with fixed teacher weights
teacher_weights = np.array(
    [0.7, -0.5, 0.9, 0.3,
     -0.4, 0.8, -0.6, 0.5]
)

teacher_output = qnn.forward(
    z.detach().numpy(),
    teacher_weights
)
target = torch.tensor(
    teacher_output.copy(), dtype=torch.float32
)

# 11. Validate forward pass
prediction = model(z)
print("Output shape:", tuple(prediction.shape))
print("Initial prediction:", prediction.detach().numpy())
print("Target:", target.numpy())

# 12. Test quantum gradients
loss = torch.mean((prediction - target) ** 2)
loss.backward()

print("Weight gradients:", model.weight.grad)
print("Input gradients:", z.grad)

assert prediction.shape == (1, 4)
assert model.weight.grad is not None
assert z.grad is not None
assert torch.isfinite(model.weight.grad).all()
assert torch.isfinite(z.grad).all()
assert torch.any(model.weight.grad.abs() > 1e-8)
assert torch.any(z.grad.abs() > 1e-8)

print("Forward and backward tests: PASSED")

# 13. Train the quantum predictor
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

print("\nTraining:")
for epoch in range(21):
    optimizer.zero_grad()
    prediction = model(z.detach())
    loss = torch.mean((prediction - target) ** 2)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d} | Loss = {loss.item():.6f}")

# 14. Final evaluation
with torch.no_grad():
    final_prediction = model(z.detach())
    final_loss = torch.mean(
        (final_prediction - target) ** 2
    )

print("\nFinal prediction:", final_prediction.numpy())
print("Target:", target.numpy())
print("Final MSE:", final_loss.item())
print("Trained parameters:", model.weight.detach().numpy())
