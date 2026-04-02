"""
Draw the VQC circuit used by cQLSTM and QuantumSSM using PennyLane's mpl drawer.

Circuit structure (for n_qubits=4, n_qlayers=1, n_esteps=1):

  Encoding block (per qubit i, input y_i):
    H → RY(arctan(y_i)) → RZ(arctan(y_i²))

  Ansatz block (repeated n_qlayers times):
    Entangling: CNOT ring (n_esteps offsets)
    Variational: RX(θ₀ᵢ) → RY(θ₁ᵢ) → RZ(θ₂ᵢ)

  Measurement:
    ⟨Z⟩ on each qubit
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── configuration ─────────────────────────────────────────────────────────────
N_QUBITS  = 4
N_QLAYERS = 1
N_ESTEPS  = 1
N_VROT    = 3   # RX, RY, RZ per qubit per layer

wires = list(range(N_QUBITS))
dev   = qml.device("default.qubit", wires=N_QUBITS)

# ── ansatz (entangling + variational) ─────────────────────────────────────────
def ansatz(params, wires_type):
    # Entangling layer — CNOT ring
    for k in range(N_ESTEPS):
        for i in range(N_QUBITS):
            qml.CNOT(wires=[wires_type[i], wires_type[(i + k + 1) % N_QUBITS]])
    # Variational layer — RX/RY/RZ
    for i in range(N_QUBITS):
        qml.RX(params[0][i], wires=wires_type[i])
        qml.RY(params[1][i], wires=wires_type[i])
        qml.RZ(params[2][i], wires=wires_type[i])

# ── full VQC ──────────────────────────────────────────────────────────────────
@qml.qnode(dev)
def vqc(inputs, weights):
    """
    inputs  : [N_QUBITS]   — post-entry linear projection y = entry(cat(x_t, h_t))
    weights : [N_QLAYERS, N_VROT, N_QUBITS]  — trainable variational params
    """
    # ── Encoding ──────────────────────────────────────────────────────────────
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)
        qml.RY(np.arctan(inputs[i]),    wires=i)   # amplitude encoding
        qml.RZ(np.arctan(inputs[i]**2), wires=i)   # phase encoding (magnitude)

    # ── Ansatz (n_qlayers repetitions) ────────────────────────────────────────
    qml.layer(ansatz, N_QLAYERS, weights, wires_type=wires)

    # ── Measurement ───────────────────────────────────────────────────────────
    return [qml.expval(qml.PauliZ(wires=i)) for i in wires]

# ── draw ──────────────────────────────────────────────────────────────────────
dummy_inputs  = np.zeros(N_QUBITS)
dummy_weights = np.zeros((N_QLAYERS, N_VROT, N_QUBITS))

fig, ax = qml.draw_mpl(
    vqc,
    decimals=None,          # hide numeric values → cleaner symbolic view
    style="pennylane",
    wire_order=wires,
)(dummy_inputs, dummy_weights)

fig.suptitle(
    "VQC — shared circuit in cQLSTM & QuantumSSM\n"
    f"({N_QUBITS} qubits · {N_QLAYERS} ansatz layer · {N_ESTEPS} CNOT offset)",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)

# ── legend explaining the three blocks ────────────────────────────────────────
legend_items = [
    mpatches.Patch(color="#4c72b0", label="Encoding:  H → RY(arctan(y)) → RZ(arctan(y²))"),
    mpatches.Patch(color="#dd8452", label="Entangling: CNOT ring  (n_esteps=1 offset)"),
    mpatches.Patch(color="#55a868", label="Variational: RX(θ₀) → RY(θ₁) → RZ(θ₂)  [trainable]"),
    mpatches.Patch(color="#c44e52", label="Measurement: ⟨Z⟩ᵢ  →  scalar ∈ [−1, 1]"),
]
ax.legend(
    handles=legend_items,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    fontsize=9,
    framealpha=0.9,
)

fig.tight_layout()
out_path = "output/vqc_circuit.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
