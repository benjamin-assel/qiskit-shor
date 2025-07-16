from qiskit.circuit import QuantumCircuit, QuantumRegister

from qiskit_shor.qft import QFTFullGate


def test_qft():
    q_reg = QuantumRegister(4)
    qc = QuantumCircuit(q_reg)
    qc.append(QFTFullGate(4), q_reg)
    expected_gate_counts = {"h": 4, "cp": 6, "swap": 2}
    actual_gate_counts = qc.decompose().count_ops()
    assert actual_gate_counts == expected_gate_counts

    # No swap
    qc = QuantumCircuit(q_reg)
    qc.append(QFTFullGate(4, do_swaps=False), q_reg)
    expected_gate_counts = {"h": 4, "cp": 6}
    actual_gate_counts = qc.decompose().count_ops()
    assert actual_gate_counts == expected_gate_counts

    # approximate QFT
    qc = QuantumCircuit(q_reg)
    qc.append(QFTFullGate(4, approximation_degree=1), q_reg)
    expected_gate_counts = {"h": 4, "cp": 5, "swap": 2}
    actual_gate_counts = qc.decompose().count_ops()
    assert actual_gate_counts == expected_gate_counts

    qc = QuantumCircuit(q_reg)
    qc.append(QFTFullGate(4, approximation_degree=2), q_reg)
    expected_gate_counts = {"h": 4, "cp": 3, "swap": 2}
    actual_gate_counts = qc.decompose().count_ops()
    assert actual_gate_counts == expected_gate_counts

    # Insert barriers
    qc = QuantumCircuit(q_reg)
    qc.append(QFTFullGate(4, insert_barriers=True), q_reg)
    expected_gate_counts = {"h": 4, "cp": 6, "swap": 2, "barrier": 4}
    actual_gate_counts = qc.decompose().count_ops()
    assert actual_gate_counts == expected_gate_counts
