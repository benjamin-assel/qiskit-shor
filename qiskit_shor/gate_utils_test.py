from qiskit.circuit import ClassicalRegister, Gate, QuantumCircuit, QuantumRegister

from qiskit_shor.gate_utils import MAJGate, MCXMAJGate, MCXUMAGate, QFTFullGate, UMAGate
from qiskit_shor.test_utils import run_simulation


def run_gate_test(gate: Gate, input_bitstr: str, expected_output_bitstr: str):
    """
    Run a test on a gate.
    """
    num_qubits = gate.num_qubits
    assert len(input_bitstr) == num_qubits, f"Expected {num_qubits} input bits, got {len(input_bitstr)}"
    assert len(expected_output_bitstr) == num_qubits, (
        f"Expected {num_qubits} output bits, got {len(expected_output_bitstr)}"
    )

    q_reg = QuantumRegister(num_qubits)
    output_reg = ClassicalRegister(num_qubits, name="output")
    qc = QuantumCircuit(q_reg, output_reg)

    input_bits = [int(bit) for bit in input_bitstr][::-1]

    for i in range(num_qubits):
        if input_bits[i] == 1:
            qc.x(q_reg[i])
    qc.append(gate, q_reg)
    qc.measure(q_reg, output_reg)

    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist.get(expected_output_bitstr) == 1, (
        f"Expected {expected_output_bitstr} for input {input_bitstr}, got dist {dist}"
    )


def test_qft_full_gate():

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


def test_maj_gate():
    run_gate_test(MAJGate(), "000", "000")
    run_gate_test(MAJGate(), "001", "001")
    run_gate_test(MAJGate(), "010", "010")
    run_gate_test(MAJGate(), "011", "111")
    run_gate_test(MAJGate(), "100", "011")
    run_gate_test(MAJGate(), "101", "110")
    run_gate_test(MAJGate(), "110", "101")
    run_gate_test(MAJGate(), "111", "100")


def test_cx_maj_gate():
    cxmaj_gate = MCXMAJGate(num_ctrl_qubits=1)
    # ctrl_bit = 1
    run_gate_test(cxmaj_gate, "0001", "0001")
    run_gate_test(cxmaj_gate, "0011", "0011")
    run_gate_test(cxmaj_gate, "0101", "0101")
    run_gate_test(cxmaj_gate, "0111", "1111")
    run_gate_test(cxmaj_gate, "1001", "0111")
    run_gate_test(cxmaj_gate, "1011", "1101")
    run_gate_test(cxmaj_gate, "1101", "1011")
    run_gate_test(cxmaj_gate, "1111", "1001")

    # ctrl_bit = 0
    run_gate_test(cxmaj_gate, "0000", "0000")
    run_gate_test(cxmaj_gate, "0010", "0010")
    run_gate_test(cxmaj_gate, "0100", "0100")
    run_gate_test(cxmaj_gate, "0110", "1110")
    run_gate_test(cxmaj_gate, "1000", "1010")
    run_gate_test(cxmaj_gate, "1010", "1000")
    run_gate_test(cxmaj_gate, "1100", "0110")
    run_gate_test(cxmaj_gate, "1110", "1100")


def test_uma_gate():
    run_gate_test(UMAGate(), "000", "000")
    run_gate_test(UMAGate(), "001", "011")
    run_gate_test(UMAGate(), "010", "010")
    run_gate_test(UMAGate(), "011", "110")
    run_gate_test(UMAGate(), "100", "111")
    run_gate_test(UMAGate(), "101", "100")
    run_gate_test(UMAGate(), "110", "101")
    run_gate_test(UMAGate(), "111", "001")


def test_cx_uma_gate():
    cxuma_gate = MCXUMAGate(num_ctrl_qubits=1)
    # ctrl_bit = 1
    run_gate_test(cxuma_gate, "0001", "0001")
    run_gate_test(cxuma_gate, "0011", "0111")
    run_gate_test(cxuma_gate, "0101", "0101")
    run_gate_test(cxuma_gate, "0111", "1101")
    run_gate_test(cxuma_gate, "1001", "1111")
    run_gate_test(cxuma_gate, "1011", "1001")
    run_gate_test(cxuma_gate, "1101", "1011")
    run_gate_test(cxuma_gate, "1111", "0011")

    # ctrl_bit = 0
    run_gate_test(cxuma_gate, "0000", "0000")
    run_gate_test(cxuma_gate, "0010", "0010")
    run_gate_test(cxuma_gate, "0100", "0100")
    run_gate_test(cxuma_gate, "0110", "1100")
    run_gate_test(cxuma_gate, "1000", "1010")
    run_gate_test(cxuma_gate, "1010", "1000")
    run_gate_test(cxuma_gate, "1100", "1110")
    run_gate_test(cxuma_gate, "1110", "0110")
