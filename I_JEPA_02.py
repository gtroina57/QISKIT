
import numpy as np
import torch

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# ---------------------------------------------------
# 1. Reproducibility
# ----------------------------------------------------

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N_QUBITS = 4
EPOCHS = 30
LEARNING_RATE = 0.05

# --------------------------------------------------
# 2. Create synthetic dataset
# --------------------------------------------------

rng = np.random.default_rng(SEED)

# 20 inputs, each containing four latent features
X = torch.tensor(
    rng.uniform(-1, 1, (20, 4)),
    dtype=torch.float32
)

# Target function with cross-feature dependencies
def target_function(x):
    return 0.3 * torch.stack([
        torch.cos(x[:, 0] + x[:, 1]),
        torch.cos(x[:, 1] - x[:, 2]),
        torch.cos(x[:, 2] + x[:, 3]),
        torch.cos(x[:, 3] - x[:, 0])
    ], dim=1)

Y = target_function(X)

# Fixed shuffled split
order = rng.permutation(len(X))
train_idx = order[:12]
test_idx = order[12:]

X_train = X[train_idx]
Y_train = Y[train_idx]

X_test = X[test_idx]
Y_test = Y[test_idx]

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

# --------------------------------------------------
# 3. Quantum predictor factory
# --------------------------------------------------

def create_quantum_predictor(entangled=True):

    x = ParameterVector("x", N_QUBITS)
    theta = ParameterVector("theta", 2 * N_QUBITS)

    qc = QuantumCircuit(N_QUBITS)

    # Input encoding
    for i in range(N_QUBITS):
        qc.ry(x[i], i)

    # First variational layer
    for i in range(N_QUBITS):
        qc.rx(theta[i], i)

    # Optional entanglement
    if entangled:
        for i in range(N_QUBITS):
            qc.cz(i, (i + 1) % N_QUBITS)

    # Second variational layer
    for i in range(N_QUBITS):
        qc.ry(theta[N_QUBITS + i], i)

    observables = []

    for i in range(N_QUBITS):
        label = (
            "I" * (N_QUBITS - i - 1)
            + "Z"
            + "I" * i
        )
        observables.append(SparsePauliOp(label))

    qnn = EstimatorQNN(
        circuit=qc,
        estimator=StatevectorEstimator(),
        observables=observables,
        input_params=list(x),
        weight_params=list(theta),
        input_gradients=True
    )

    # Same initial parameters for both models
    model = TorchConnector(
        qnn,
        initial_weights=np.full(8, 0.25)
    )

    return model, qc

# --------------------------------------------------
# 4. Training function
# --------------------------------------------------

def train_model(model, name):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []

    print("\nTraining:", name)

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        prediction = model(X_train)

        loss = torch.mean(
            (prediction - Y_train) ** 2
        )

        loss.backward()
        optimizer.step()

        history.append(loss.item())

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch:02d} "
                f"| Train MSE: {loss.item():.6f}"
            )

    return history

# --------------------------------------------------
# 5. Create both models
# --------------------------------------------------

model_ent, circuit_ent = create_quantum_predictor(
    entangled=True
)

model_local, circuit_local = create_quantum_predictor(
    entangled=False
)

print("\nENTANGLED CIRCUIT")
print(circuit_ent.draw("text"))

print("\nNON-ENTANGLED CIRCUIT")
print(circuit_local.draw("text"))

# --------------------------------------------------
# 6. Train
# --------------------------------------------------

history_ent = train_model(
    model_ent,
    "Entangled predictor"
)

history_local = train_model(
    model_local,
    "Non-entangled predictor"
)

# --------------------------------------------------
# 7. Evaluation
# --------------------------------------------------

def evaluate(model, inputs, targets):
    with torch.no_grad():
        predictions = model(inputs)
        mse = torch.mean(
            (predictions - targets) ** 2
        )
    return mse.item(), predictions

ent_train, _ = evaluate(
    model_ent, X_train, Y_train
)

ent_test, ent_predictions = evaluate(
    model_ent, X_test, Y_test
)

local_train, _ = evaluate(
    model_local, X_train, Y_train
)

local_test, local_predictions = evaluate(
    model_local, X_test, Y_test
)

print("\n" + "=" * 45)
print("FINAL RESULTS")
print("=" * 45)

print(f"Entangled - Train MSE: {ent_train:.6f}")
print(f"Entangled - Test MSE:  {ent_test:.6f}")

print(f"Local     - Train MSE: {local_train:.6f}")
print(f"Local     - Test MSE:  {local_test:.6f}")

print("\nSample test prediction:")
print("Target:   ", Y_test[0].numpy())
print("Entangled:", ent_predictions[0].numpy())
print("Local:    ", local_predictions[0].numpy())

# --------------------------------------------------
# 8. Check gradients through the predictor
# --------------------------------------------------

probe = X_test[:1].clone().detach()
probe.requires_grad_(True)

model_ent.zero_grad()
probe_output = model_ent(probe)
probe_loss = torch.mean(
    (probe_output - Y_test[:1]) ** 2
)

probe_loss.backward()

print("\nInput gradients:", probe.grad)
print("Weight gradients:", model_ent.weight.grad)

assert probe.grad is not None
assert model_ent.weight.grad is not None
assert torch.isfinite(probe.grad).all()
assert torch.isfinite(model_ent.weight.grad).all()

print("\nGradient checks passed.")
