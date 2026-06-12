import math

from qiskit.circuit import ClassicalRegister, QuantumRegister

from qiskit_shor.rc_adder import RCAdderCircuit
from qiskit_shor.test_utils import run_simulation


def test_bitwise_AND_classical() -> None:
    x_reg = QuantumRegister(3)
    output_reg = ClassicalRegister(3, name="output")

    # 1 = 1 mod 8 = '001'
    qc = RCAdderCircuit(x_reg, output_reg)
    qc.bitwise_AND_classical(1, x_reg)
    qc.measure(x_reg, output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["001"] == 1

    # 14 = 6 mod 8 = '110'
    qc = RCAdderCircuit(x_reg, output_reg)
    qc.bitwise_AND_classical(14, x_reg)
    qc.measure(x_reg, output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["110"] == 1


def test_add_quantum() -> None:
    x_reg = QuantumRegister(3)
    y_reg = QuantumRegister(3)
    a_reg = QuantumRegister(1)
    output_reg = ClassicalRegister(3, name="output")

    # 2 + 3 = 5 mod 8 = '101' (no overflow bit)
    qc = RCAdderCircuit(x_reg, y_reg, a_reg, output_reg)
    qc.bitwise_AND_classical(2, x_reg)  # encode 2 in x register
    qc.bitwise_AND_classical(3, y_reg)  # encode 3 in y register
    qc.add_quantum(x_reg, y_reg, a_reg[0])
    qc.measure(y_reg, output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["101"] == 1

    # 4 + 7 = 3 mod 8 = '011' (no overflow bit)
    qc = RCAdderCircuit(x_reg, y_reg, a_reg, output_reg)
    qc.bitwise_AND_classical(4, x_reg)
    qc.bitwise_AND_classical(7, y_reg)
    qc.add_quantum(x_reg, y_reg, a_reg[0])
    qc.measure(y_reg, output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["011"] == 1

    # 4 + 7 = 11 = '1011' (with overflow bit)
    o_reg = QuantumRegister(1)
    output_reg = ClassicalRegister(4, name="output")
    qc = RCAdderCircuit(x_reg, y_reg, a_reg, o_reg, output_reg)
    qc.bitwise_AND_classical(4, x_reg)
    qc.bitwise_AND_classical(7, y_reg)
    qc.add_quantum(x_reg, y_reg, a_reg[0], overflow_bit=o_reg[0])
    qc.measure(y_reg[:] + o_reg[:], output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["1011"] == 1


def test_c_add_quantum() -> None:
    c_reg = QuantumRegister(1)
    x_reg = QuantumRegister(3)
    y_reg = QuantumRegister(3)
    a_reg = QuantumRegister(1)
    output_reg = ClassicalRegister(3, name="output")

    #  Control bit = |0>, no addition
    # 3 = 3 mod 8 = '011'
    qc = RCAdderCircuit(c_reg, x_reg, y_reg, a_reg, output_reg)
    qc.bitwise_AND_classical(2, x_reg)
    qc.bitwise_AND_classical(3, y_reg)
    qc.c_add_quantum(c_reg[0], x_reg, y_reg, a_reg[0])
    qc.measure(y_reg, output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["011"] == 1

    #  Control bit = |1>, addition
    # 2 + 3 = 5 mod 8 = '101'
    qc = RCAdderCircuit(c_reg, x_reg, y_reg, a_reg, output_reg)
    qc.x(c_reg[0])
    qc.bitwise_AND_classical(2, x_reg)
    qc.bitwise_AND_classical(3, y_reg)
    qc.c_add_quantum(c_reg[0], x_reg, y_reg, a_reg[0])
    qc.measure(y_reg, output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["101"] == 1


def test_add_classical() -> None:
    y_reg = QuantumRegister(3)
    a_reg = QuantumRegister(4)
    output_reg = ClassicalRegister(3, name="output")

    # 2 + 4 = 6 mod 8 = '110'
    qc = RCAdderCircuit(y_reg, a_reg, output_reg)
    qc.bitwise_AND_classical(2, y_reg)
    qc.add_classical(4, y_reg, a_reg)
    qc.measure(y_reg, output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["110"] == 1


def test_c_add_classical() -> None:
    c_reg = QuantumRegister(1)
    y_reg = QuantumRegister(3)
    a_reg = QuantumRegister(4)
    output_reg = ClassicalRegister(3, name="output")

    # Control bit = |0>, addition disabled
    # 2 = 2 mod 8 = '010'
    qc = RCAdderCircuit(c_reg, y_reg, a_reg, output_reg)
    qc.bitwise_AND_classical(2, y_reg)
    qc.c_add_classical(c_reg, 4, y_reg, a_reg)
    qc.measure(y_reg, output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["010"] == 1

    # Control bit = |1>, addition enabled
    # 2 + 4 = 6 mod 8 = '110'
    qc = RCAdderCircuit(c_reg, y_reg, a_reg, output_reg)
    qc.x(c_reg[0])
    qc.bitwise_AND_classical(2, y_reg)
    qc.c_add_classical(c_reg, 4, y_reg, a_reg)
    qc.measure(y_reg, output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["110"] == 1


def test_add_classical_modulo() -> None:
    y_reg = QuantumRegister(3)
    output_reg = ClassicalRegister(4, name="output")
    anc_reg = QuantumRegister(1)
    ov_reg = QuantumRegister(1)
    a_reg = QuantumRegister(5)
    anc_ouput_reg = ClassicalRegister(1, name="anc_output")

    # With ancilla reset
    qc = RCAdderCircuit(y_reg, anc_reg, ov_reg, a_reg, output_reg, anc_ouput_reg)
    # 0 + 3 + 2 = 5 mod 7 = '0101'
    qc.add_classical(3, y_reg, a_reg)
    qc.add_classical_modulo(
        X=2, y_reg=y_reg, ancilla_bit=anc_reg[0], overflow_bit=ov_reg[0], a_reg=a_reg, N=7
    )
    qc.measure(y_reg[:] + ov_reg[:], output_reg)
    qc.measure(anc_reg, anc_ouput_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    anc_dist = res.data.anc_output.get_counts()

    assert dist["0101"] == 1
    assert anc_dist["0"] == 1

    # Without ancilla reset
    qc = RCAdderCircuit(y_reg, anc_reg, ov_reg, a_reg, output_reg, anc_ouput_reg)
    # 0 + 3 + 2 = 5 mod 7 = '0101'
    qc.add_classical(3, y_reg, a_reg)
    qc.add_classical_modulo(
        X=2,
        y_reg=y_reg,
        ancilla_bit=anc_reg[0],
        overflow_bit=ov_reg[0],
        N=7,
        a_reg=a_reg,
        reset_ancilla=False,
    )
    qc.measure(y_reg[:] + ov_reg[:], output_reg)
    qc.measure(anc_reg, anc_ouput_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    anc_dist = res.data.anc_output.get_counts()

    assert dist["0101"] == 1
    assert anc_dist["1"] == 1


def test_c_add_classical_modulo() -> None:
    y_reg = QuantumRegister(3)
    output_reg = ClassicalRegister(4, name="output")
    c_reg = QuantumRegister(1)
    anc_reg = QuantumRegister(1)
    ov_reg = QuantumRegister(1)
    a_reg = QuantumRegister(5)
    anc_ouput_reg = ClassicalRegister(1, name="anc_output")

    # Perform modulo 7 operations.
    N = 7

    # Control bit = |0>
    qc = RCAdderCircuit(c_reg, y_reg, output_reg, anc_reg, ov_reg, a_reg, anc_ouput_reg)
    # Add 3 to y register
    qc.add_classical(3, y_reg, a_reg)
    # Control_add 2 to y register
    qc.c_add_classical_modulo(c_reg, 2, y_reg, anc_reg[0], ov_reg[0], N, a_reg)
    qc.measure(y_reg[:] + ov_reg[:], output_reg)
    qc.measure(anc_reg, anc_ouput_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    a_dist = res.data.anc_output.get_counts()

    assert dist["0011"] == 1
    assert a_dist["0"] == 1

    # Control bit = |1>
    qc = RCAdderCircuit(c_reg, y_reg, output_reg, anc_reg, ov_reg, a_reg, anc_ouput_reg)
    qc.x(c_reg[0])
    # Add 3 to y register
    qc.add_classical(3, y_reg, a_reg)
    # Control_add 2 to y register
    qc.c_add_classical_modulo(c_reg, 2, y_reg, anc_reg[0], ov_reg[0], N, a_reg)
    qc.measure(y_reg[:] + ov_reg[:], output_reg)
    qc.measure(anc_reg, anc_ouput_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    a_dist = res.data.anc_output.get_counts()

    assert dist["0101"] == 1
    assert a_dist["0"] == 1


def test_add_quantum_modulo() -> None:
    x_reg = QuantumRegister(3)
    y_reg = QuantumRegister(3)
    outpout_reg = ClassicalRegister(4, name="output")
    anc_reg = QuantumRegister(1)
    ov_reg = QuantumRegister(1)
    a_reg = QuantumRegister(5)
    anc_ouput_reg = ClassicalRegister(1, name="anc_output")
    qc = RCAdderCircuit(x_reg, y_reg, outpout_reg, anc_reg, ov_reg, a_reg, anc_ouput_reg)

    # Perform modulo 7 operations.
    N = 7
    # Add 4 in x register
    qc.add_classical(4, x_reg, a_reg)
    # Add 6 in y register
    qc.add_classical(6, y_reg, a_reg)
    # Add 10 times x register to y register modulo 7
    qc.add_quantum_modulo(x_reg, y_reg, anc_reg[0], ov_reg[0], N, a_reg, A=10)
    qc.measure(y_reg[:] + ov_reg[:], outpout_reg)
    qc.measure(anc_reg, anc_ouput_reg)

    # Expected result = 10*4 + 6 = 4 mod 7 = "0100"
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    a_dist = res.data.anc_output.get_counts()
    assert dist["0100"] == 1
    assert a_dist["0"] == 1


def test_c_add_quantum_modulo() -> None:
    control_reg = QuantumRegister(1)
    x_reg = QuantumRegister(3)
    y_reg = QuantumRegister(3)
    outpout_reg = ClassicalRegister(4, name="output")
    anc_reg = QuantumRegister(1)
    ov_reg = QuantumRegister(1)
    a_reg = QuantumRegister(5)
    anc_ouput_reg = ClassicalRegister(1, name="anc_output")

    # Perform modulo 7 operations.
    N = 7

    # Case 1: Control bit = |0>
    qc = RCAdderCircuit(control_reg, x_reg, y_reg, outpout_reg, anc_reg, ov_reg, a_reg, anc_ouput_reg)
    # Add 4 in x register
    qc.add_classical(4, x_reg, a_reg)
    # Add 6 in y register
    qc.add_classical(6, y_reg, a_reg)
    # Add 10 times x register to y register modulo 7
    qc.c_add_quantum_modulo(control_reg, x_reg, y_reg, anc_reg[0], ov_reg[0], N, a_reg, A=10)
    qc.measure(y_reg[:] + ov_reg[:], outpout_reg)
    qc.measure(anc_reg, anc_ouput_reg)
    # Expected result = 6 mod 7 = "0110"
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    a_dist = res.data.anc_output.get_counts()
    assert dist["0110"] == 1
    assert a_dist["0"] == 1

    # Case 2: Control bit = |1>
    qc = RCAdderCircuit(control_reg, x_reg, y_reg, outpout_reg, anc_reg, ov_reg, a_reg, anc_ouput_reg)
    qc.x(control_reg[0])
    # Add 4 in x register
    qc.add_classical(4, x_reg, a_reg)
    # Add 6 in y register
    qc.add_classical(6, y_reg, a_reg)
    # Add 10 times x register to y register modulo 7
    qc.c_add_quantum_modulo(control_reg, x_reg, y_reg, anc_reg[0], ov_reg[0], N, a_reg, A=10)
    qc.measure(y_reg[:] + ov_reg[:], outpout_reg)
    qc.measure(anc_reg, anc_ouput_reg)
    # Expected result = 10*4 + 6 = 4 mod 7 = "0100"
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    a_dist = res.data.anc_output.get_counts()

    assert dist["0100"] == 1
    assert a_dist["0"] == 1


def test_multiply_modulo() -> None:
    # Perform modulo 5 operations.
    N = 5
    n = math.ceil(math.log2(N))  # = 3
    x_reg = QuantumRegister(n)
    y_reg = QuantumRegister(n)
    ov_reg = QuantumRegister(1)
    a_reg = QuantumRegister(n + 2)
    anc_reg = QuantumRegister(1)
    outpout_reg = ClassicalRegister(2 * n + 2, name="output")
    o_bit = ov_reg[0]
    a_bit = anc_reg[0]

    # Case 1: with uncomputation, with swap
    qc = RCAdderCircuit(x_reg, y_reg, outpout_reg, ov_reg, a_reg, anc_reg)
    # Add 3 in x register
    qc.add_classical(3, x_reg, a_reg)
    # |3> ->  |9*3 modulo 5>
    qc.multiply_modulo(A=9, x_reg=x_reg, y_reg=y_reg, overflow_bit=o_bit, ancilla_bit=a_bit, N=N, a_reg=a_reg)
    qc.measure(x_reg[:] + y_reg[:] + ov_reg[:] + anc_reg[:], outpout_reg)
    # Expected x_reg value = 9*3 mod 5 = 2 = "010" -> Less significant output bits
    # Expected y_reg value = "000" -> Middle output bits
    # Expected ancilla = "00" -> More significant output bits
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["00000010"] == 1

    # Case 2: with uncomputation, without swap
    qc = RCAdderCircuit(x_reg, y_reg, outpout_reg, ov_reg, a_reg, anc_reg)
    # Add 3 in x register
    qc.add_classical(3, x_reg, a_reg)
    # |3> ->  |9*3 modulo 5>
    qc.multiply_modulo(
        A=9,
        x_reg=x_reg,
        y_reg=y_reg,
        overflow_bit=o_bit,
        ancilla_bit=a_bit,
        N=N,
        a_reg=a_reg,
        with_swap=False,
    )
    qc.measure(x_reg[:] + y_reg[:] + ov_reg[:] + anc_reg[:], outpout_reg)
    # Expected x_reg value = "000"
    # Expected y_reg value = 9*3 mod 5 = 2 = "010"
    # Expected ancilla = "00"
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["00010000"] == 1

    # Case 3: without uncomputation, with swap
    qc = RCAdderCircuit(x_reg, y_reg, outpout_reg, ov_reg, a_reg, anc_reg)
    # Add 3 in x register
    qc.add_classical(3, x_reg, a_reg)
    # |3> ->  |9*3 modulo 5>
    qc.multiply_modulo(
        A=9,
        x_reg=x_reg,
        y_reg=y_reg,
        overflow_bit=o_bit,
        ancilla_bit=a_bit,
        N=N,
        a_reg=a_reg,
        with_uncomputation=False,
    )
    qc.measure(x_reg[:] + y_reg[:] + ov_reg[:] + anc_reg[:], outpout_reg)
    # Expected x_reg value = 9*3 mod 5 = 2 = "010" -> Less significant output bits
    # Expected y_reg value = 3 ="011" -> Middle output bits
    # Expected ancilla = "00" -> More significant output bits
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["00011010"] == 1

    # Case 4: without uncomputation, without swap
    qc = RCAdderCircuit(x_reg, y_reg, outpout_reg, ov_reg, a_reg, anc_reg)
    # Add 3 in x register
    qc.add_classical(3, x_reg, a_reg)
    # |3> ->  |9*3 modulo 5>
    qc.multiply_modulo(
        A=9,
        x_reg=x_reg,
        y_reg=y_reg,
        overflow_bit=o_bit,
        ancilla_bit=a_bit,
        N=N,
        a_reg=a_reg,
        with_swap=False,
        with_uncomputation=False,
    )
    qc.measure(x_reg[:] + y_reg[:] + ov_reg[:] + anc_reg[:], outpout_reg)
    # Expected x_reg value = 3 = "011"
    # Expected y_reg value = 9*3 mod 5 = 2 = "010"
    # Expected ancilla = "00"
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["00010011"] == 1


def test_c_multiply_modulo() -> None:
    # Perform modulo 5 operations.
    N = 5
    n = math.ceil(math.log2(N))  # = 3
    control_reg = QuantumRegister(2)
    x_reg = QuantumRegister(n)
    y_reg = QuantumRegister(n)
    ov_reg = QuantumRegister(1)
    a_reg = QuantumRegister(n + 2)
    anc_reg = QuantumRegister(1)
    outpout_reg = ClassicalRegister(2 * n + 2, name="output")
    o_bit = ov_reg[0]
    a_bit = anc_reg[0]

    # Case 1: control bits = |01> (no operation)
    qc = RCAdderCircuit(control_reg, x_reg, y_reg, outpout_reg, ov_reg, a_reg, anc_reg)
    qc.x(control_reg[1])
    # Add 3 = "011" in x register
    qc.add_classical(3, x_reg, a_reg)
    qc.c_multiply_modulo(
        control_reg=control_reg,
        A=9,
        x_reg=x_reg,
        y_reg=y_reg,
        overflow_bit=o_bit,
        a_reg=a_reg,
        ancilla_bit=a_bit,
        N=N,
    )
    qc.measure(x_reg[:] + y_reg[:] + ov_reg[:] + anc_reg[:], outpout_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    print(dist)
    assert dist["00000011"] == 1

    # Case 2: control bits = |11>
    qc = RCAdderCircuit(control_reg, x_reg, y_reg, outpout_reg, ov_reg, a_reg, anc_reg)
    qc.x(control_reg[0])
    qc.x(control_reg[1])
    # Add 3 = "011" in x register
    qc.add_classical(3, x_reg, a_reg)
    qc.c_multiply_modulo(
        control_reg=control_reg,
        A=9,
        x_reg=x_reg,
        y_reg=y_reg,
        overflow_bit=o_bit,
        a_reg=a_reg,
        ancilla_bit=a_bit,
        N=N,
    )
    qc.measure(x_reg[:] + y_reg[:] + ov_reg[:] + anc_reg[:], outpout_reg)
    # Expected x_reg value = 9*3 mod 5 = 2 = "010" -> Less significant output bits
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["00000010"] == 1


def test_exponentiate_modulo() -> None:
    # Perform modulo 5 operations.
    N = 5
    n = math.ceil(math.log2(N))  # = 3
    m = 3
    x_reg = QuantumRegister(m)
    y_reg = QuantumRegister(n)
    ancilla_reg = QuantumRegister(n + 2)
    a_reg = QuantumRegister(n + 2)
    outpout_reg = ClassicalRegister(m + 2 * n + 2, name="output")

    qc = RCAdderCircuit(x_reg, y_reg, ancilla_reg, a_reg, outpout_reg)
    qc.add_classical(4, x_reg, a_reg)
    qc.add_classical(3, y_reg, a_reg)
    # |4> |3> |0> ->  |4> |2^4 * 3 modulo 5> |0>
    qc.exponentiate_modulo(A=2, x_reg=x_reg, y_reg=y_reg, ancilla_reg=ancilla_reg, N=N, a_reg=a_reg)
    qc.measure(x_reg[:] + y_reg[:] + ancilla_reg[:], outpout_reg)
    # Expected x_reg value = 4 = "100"
    # Expected y_reg value = 2^4 * 3 mod 5 = 3 = "011"
    # Expected ancilla = "00000"
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["00000" + "011" + "100"] == 1

    qc = RCAdderCircuit(x_reg, y_reg, ancilla_reg, a_reg, outpout_reg)
    qc.add_classical(7, x_reg, a_reg)
    qc.add_classical(1, y_reg, a_reg)
    # |7> |1> |0> ->  |7> |2^7 * 1 modulo 5> |0>
    qc.exponentiate_modulo(A=2, x_reg=x_reg, y_reg=y_reg, ancilla_reg=ancilla_reg, N=N, a_reg=a_reg)
    qc.measure(x_reg[:] + y_reg[:] + ancilla_reg[:], outpout_reg)
    # Expected x_reg value = 7 = "111"
    # Expected y_reg value = 2^7 * 1 modulo 5 = 3 = "011"
    # Expected ancilla = "00000"
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    print(dist)
    assert dist["00000" + "011" + "111"] == 1
