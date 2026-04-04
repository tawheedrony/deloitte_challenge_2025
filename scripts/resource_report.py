"""
scripts/resource_report.py  —  Task 1A quantum resource requirements
=====================================================================
Generates a resource report for all quantum models: circuit depth, gate
counts (single-qubit, two-qubit), number of trainable parameters, and
single-forward-pass wall-clock time.

Usage:
    python scripts/resource_report.py
    python scripts/resource_report.py --n_qubits 8
    python scripts/resource_report.py --out output/resource_report.txt

Output:
    Printed report + optional file save with:
    • Per-model circuit specs (depth, 1q gates, 2q gates)
    • Parameter counts (quantum vs classical)
    • Timing benchmarks (forward pass on batch of 32)
"""

import argparse
import sys
import time
import warnings
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pennylane as qml
import torch


# ─────────────────────────── model builders ──────────────────────────────

def build_quantum_models(input_size: int, hidden_size: int,
                         n_qubits: int, n_qlayers: int, n_esteps: int):
    """Return dict of {name: model} for all quantum architectures."""
    models = {}

    from models.cQLSTM import cQLSTM
    models["cQLSTM"] = cQLSTM(
        input_size=input_size, hidden_size=hidden_size,
        n_qubits=n_qubits, n_qlayers=n_qlayers, n_esteps=n_esteps,
    )

    from models.QLSTM import QLSTM
    models["QLSTM"] = QLSTM(
        input_size=input_size, hidden_size=hidden_size,
        n_qubits=n_qubits, n_qlayers=n_qlayers,
    )

    from models.QSSM import QuantumSSM
    models["QSSM"] = QuantumSSM(
        input_size=input_size, hidden_size=hidden_size,
        n_qubits=n_qubits, n_qlayers=n_qlayers, n_esteps=n_esteps,
    )

    from models.cQSSM import cQSSM
    models["cQSSM"] = cQSSM(
        input_size=input_size, hidden_size=hidden_size,
        n_qubits=n_qubits, n_qlayers=n_qlayers, n_esteps=n_esteps,
    )

    from models.QDeltaNet import QDeltaNet
    models["QDeltaNet"] = QDeltaNet(
        input_size=input_size, hidden_size=hidden_size,
        n_qubits=n_qubits, n_qlayers=n_qlayers, n_esteps=n_esteps,
    )

    return models


def build_classical_models(input_size: int, hidden_size: int):
    """Return dict of {name: model} for classical baselines."""
    models = {}

    from models.LSTM import LSTM
    models["LSTM"] = LSTM(input_size=input_size, hidden_size=hidden_size)

    from models.cLSTM import cLSTM
    models["cLSTM"] = cLSTM(input_size=input_size, hidden_size=hidden_size)

    from models.SSM import SSM
    models["SSM"] = SSM(input_size=input_size, hidden_size=hidden_size)

    from models.DeltaNet import DeltaNet
    models["DeltaNet"] = DeltaNet(input_size=input_size, hidden_size=hidden_size)

    return models


# ─────────────────────────── circuit specs ───────────────────────────────

def get_circuit_specs(n_qubits: int, n_qlayers: int, n_esteps: int):
    """
    Build a standalone VQC matching the cQLSTM design and extract specs
    using qml.specs.
    """
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        features = inputs.transpose(1, 0)
        ry_params = [torch.arctan(f) for f in features]
        rz_params = [torch.arctan(f ** 2) for f in features]
        for i in range(n_qubits):
            qml.Hadamard(wires=wires[i])
            qml.RY(ry_params[i], wires=wires[i])
            qml.RZ(rz_params[i], wires=wires[i])

        for _ in range(n_qlayers):
            for k in range(n_esteps):
                for i in range(n_qubits):
                    qml.CNOT(wires=[wires[i], wires[(i + k + 1) % n_qubits]])
            for i in range(n_qubits):
                qml.RX(weights[0][i], wires=wires[i])
                qml.RY(weights[1][i], wires=wires[i])
                qml.RZ(weights[2][i], wires=wires[i])

        return [qml.expval(qml.PauliZ(wires=i)) for i in wires]

    dummy_inputs = torch.randn(1, n_qubits)
    dummy_weights = torch.randn(3, n_qubits)
    raw_specs = qml.specs(circuit)(dummy_inputs, dummy_weights)

    # Extract into a plain dict from CircuitSpecs object
    res = raw_specs.resources
    # trainable params = n_qlayers * n_vrotations * n_qubits
    n_trainable = n_qlayers * 3 * n_qubits
    specs = {
        "num_device_wires": raw_specs.num_device_wires,
        "depth": res.depth,
        "num_gates": res.num_gates,
        "gate_types": dict(res.gate_types),
        "num_trainable_params": n_trainable,
    }

    return specs


# ─────────────────────────── timing ──────────────────────────────────────

def time_forward(model, input_size, seq_len=3, batch_size=32, n_runs=5):
    """Time the forward pass (CPU, no grad) averaged over n_runs."""
    x = torch.randn(batch_size, seq_len, input_size)
    model.eval()

    # warmup
    with torch.no_grad():
        model(x)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        times.append(time.perf_counter() - t0)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
    }


# ─────────────────────────── parameter counting ─────────────────────────

def count_params(model):
    """Count total, trainable, and quantum-specific parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # quantum params: look for VQC TorchLayer weights
    quantum = 0
    for name, p in model.named_parameters():
        if "VQC" in name or "qnode" in name.lower():
            quantum += p.numel()

    return {"total": total, "trainable": trainable, "quantum": quantum,
            "classical": total - quantum}


# ─────────────────────────── report ──────────────────────────────────────

def generate_report(input_size: int, hidden_size: int,
                    n_qubits: int, n_qlayers: int, n_esteps: int):
    buf = StringIO()
    w = 72

    def pr(s=""):
        buf.write(s + "\n")

    pr("=" * w)
    pr("  QUANTUM RESOURCE REQUIREMENTS REPORT")
    pr("  Task 1A — 2026 Quantum Sustainability Challenge")
    pr("=" * w)
    pr()

    # ── Circuit specs ─────────────────────────────────────────────────
    pr("1. VARIATIONAL QUANTUM CIRCUIT (VQC) SPECIFICATIONS")
    pr("-" * w)
    specs = get_circuit_specs(n_qubits, n_qlayers, n_esteps)

    pr(f"   Qubits              : {n_qubits}")
    pr(f"   Variational layers  : {n_qlayers}")
    pr(f"   Entangling steps    : {n_esteps}")
    pr(f"   Circuit depth       : {specs['depth']}")
    pr(f"   Total wires         : {specs['num_device_wires']}")
    pr(f"   Total gates         : {specs['num_gates']}")
    pr(f"   Gate types          : {specs['gate_types']}")

    # count 1q vs 2q gates
    gate_types = specs["gate_types"]
    two_q_gates = {"CNOT", "CZ", "SWAP", "CRX", "CRY", "CRZ"}
    n_1q = sum(v for k, v in gate_types.items() if k not in two_q_gates)
    n_2q = sum(v for k, v in gate_types.items() if k in two_q_gates)
    pr(f"   Single-qubit gates  : {n_1q}")
    pr(f"   Two-qubit gates     : {n_2q}")

    num_params = specs.get("num_trainable_params", "N/A")
    pr(f"   Trainable VQC params: {num_params}")
    pr(f"   Simulator           : default.qubit (PennyLane)")
    pr(f"   Diff method         : backprop")
    pr()

    pr("   Encoding (per qubit i):")
    pr("     H  ->  RY(arctan(x_i))  ->  RZ(arctan(x_i^2))")
    pr()
    pr("   Ansatz (repeated n_qlayers times):")
    pr("     CNOT ring (n_esteps offsets)  ->  RX/RY/RZ variational rotations")
    pr()
    pr("   Measurement:")
    pr("     <Z> expectation value per qubit")
    pr()

    # ── Parameter counts ──────────────────────────────────────────────
    pr("2. PARAMETER COUNTS")
    pr("-" * w)
    pr(f"   {'Model':<12} {'Total':>8} {'Quantum':>8} {'Classical':>10} {'Type':<20}")
    pr(f"   {'─'*12} {'─'*8} {'─'*8} {'─'*10} {'─'*20}")

    quantum_models = build_quantum_models(input_size, hidden_size,
                                          n_qubits, n_qlayers, n_esteps)
    classical_models = build_classical_models(input_size, hidden_size)

    all_params = {}
    for name, model in quantum_models.items():
        p = count_params(model)
        all_params[name] = p
        pr(f"   {name:<12} {p['total']:>8,} {p['quantum']:>8,} {p['classical']:>10,} {'Quantum hybrid':<20}")

    for name, model in classical_models.items():
        p = count_params(model)
        all_params[name] = p
        pr(f"   {name:<12} {p['total']:>8,} {p['quantum']:>8,} {p['classical']:>10,} {'Classical baseline':<20}")

    pr()

    # ── Timing benchmarks ─────────────────────────────────────────────
    pr("3. TIMING BENCHMARKS (forward pass, batch=32, seq=3, CPU)")
    pr("-" * w)
    pr(f"   {'Model':<12} {'Mean (ms)':>10} {'Std (ms)':>10} {'Min (ms)':>10} {'Speedup vs cQLSTM':>18}")
    pr(f"   {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*18}")

    all_times = {}
    for name, model in {**quantum_models, **classical_models}.items():
        t = time_forward(model, input_size)
        all_times[name] = t

    ref_time = all_times.get("cQLSTM", {}).get("mean_ms", 1)
    for name, t in all_times.items():
        speedup = ref_time / t["mean_ms"] if t["mean_ms"] > 0 else float("inf")
        sp_str = f"{speedup:.1f}x" if name != "cQLSTM" else "1.0x (ref)"
        pr(f"   {name:<12} {t['mean_ms']:>10.1f} {t['std_ms']:>10.1f} {t['min_ms']:>10.1f} {sp_str:>18}")

    pr()

    # ── VQC count per model ───────────────────────────────────────────
    pr("4. VQC USAGE PER MODEL")
    pr("-" * w)
    vqc_info = {
        "cQLSTM":    f"1 x {n_qubits}-qubit VQC (unified, all 4 gates)",
        "QLSTM":     f"4 x {n_qubits}-qubit VQC (one per gate: f, i, g, o)",
        "QSSM":      f"1 x {n_qubits}-qubit VQC (gating + delta)",
        "cQSSM":     f"1 x {n_qubits}-qubit VQC (gating + delta + decay)",
        "QDeltaNet": f"3 x {n_qubits}-qubit VQC (Q, K, V projections)",
    }
    for name, desc in vqc_info.items():
        pr(f"   {name:<12} {desc}")

    pr()

    # ── Summary ───────────────────────────────────────────────────────
    pr("5. SUMMARY")
    pr("-" * w)
    best_q = min(quantum_models.keys(), key=lambda n: all_params[n]["total"])
    pr(f"   Most parameter-efficient quantum model : {best_q} ({all_params[best_q]['total']:,} params)")
    pr(f"   Classical LSTM baseline                : LSTM ({all_params.get('LSTM', {}).get('total', 0):,} params)")
    ratio = all_params.get("LSTM", {}).get("total", 1) / max(all_params.get(best_q, {}).get("total", 1), 1)
    pr(f"   Parameter reduction factor             : {ratio:.1f}x fewer params in {best_q}")
    pr()
    pr(f"   Quantum advantage: VQC provides exponentially compact feature")
    pr(f"   encoding in {n_qubits}-qubit Hilbert space (dim 2^{n_qubits} = {2**n_qubits}),")
    pr(f"   while using only {n_qlayers * 3 * n_qubits} trainable rotation parameters")
    pr(f"   per variational layer.")
    pr()
    pr("=" * w)

    return buf.getvalue()


# ─────────────────────────── main ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate quantum resource requirements report for Task 1A.")
    parser.add_argument("--input_size", type=int, default=15,
                        help="Input feature count (default: 15 for engineered dataset)")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--n_qlayers", type=int, default=1)
    parser.add_argument("--n_esteps", type=int, default=1)
    parser.add_argument("--out", type=str, default=None,
                        help="Save report to file (default: print to stdout)")
    args = parser.parse_args()

    report = generate_report(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        n_qubits=args.n_qubits,
        n_qlayers=args.n_qlayers,
        n_esteps=args.n_esteps,
    )

    print(report)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report)
        print(f"\nSaved  ->  {args.out}")


if __name__ == "__main__":
    main()
