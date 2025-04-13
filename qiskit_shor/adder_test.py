import math

from qiskit.circuit import ClassicalRegister, QuantumRegister
from qiskit.primitives import StatevectorSampler
from qiskit.primitives.containers.sampler_pub_result import SamplerPubResult
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_shor.adder import AdderCircuit


def run_simulation(qc, num_shots: int = 1) -> SamplerPubResult:
    pm = generate_preset_pass_manager(optimization_level=1)
    qc_isa = pm.run(qc)
    sampler = StatevectorSampler()
    result = sampler.run([qc_isa], shots=num_shots).result()[0]

    return result


def test_add_classical() -> None:
    y_reg = QuantumRegister(3)
    output_reg = ClassicalRegister(3, name="output")

    # 0 + 3 = 3 mod 8 = '011'
    qc = AdderCircuit(y_reg, output_reg)
    qc.add_classical(3, y_reg)
    qc.measure(y_reg, output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["011"] == 1

    # 0 + 23 + (-3) = 4 mod 8 = '100'
    qc = AdderCircuit(y_reg, output_reg)
    qc.add_classical(23, y_reg)
    qc.add_classical(-3, y_reg)
    qc.measure(y_reg, output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["100"] == 1


def test_c_add_classical() -> None:
    y_reg = QuantumRegister(3)
    output_reg = ClassicalRegister(3, name="output")
    c_reg = QuantumRegister(2)

    # Control bit = |00>
    qc = AdderCircuit(c_reg, y_reg, output_reg)
    qc.c_add_classical(c_reg[:], 3, y_reg)
    qc.measure(y_reg, output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["000"] == 1

    # Control bit = |10>
    qc = AdderCircuit(c_reg, y_reg, output_reg)
    qc.x(c_reg[0])
    qc.c_add_classical(c_reg, 3, y_reg)
    qc.measure(y_reg, output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["000"] == 1

    # Control bit = |01>
    qc = AdderCircuit(c_reg, y_reg, output_reg)
    qc.x(c_reg[1])
    qc.c_add_classical(c_reg, 3, y_reg)
    qc.measure(y_reg, output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["000"] == 1

    # Control bit = |11>
    qc = AdderCircuit(c_reg, y_reg, output_reg)
    qc.x(c_reg[0])
    qc.x(c_reg[1])
    qc.c_add_classical(c_reg, 3, y_reg)
    qc.measure(y_reg, output_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["011"] == 1


def test_add_classical_modulo() -> None:
    y_reg = QuantumRegister(4)
    output_reg = ClassicalRegister(4, name="output")
    anc_reg = QuantumRegister(1)
    anc_ouput_reg = ClassicalRegister(1, name="anc_output")

    # With ancilla clean up
    qc = AdderCircuit(y_reg, output_reg, anc_reg, anc_ouput_reg)
    # 0 + 3 + 2 = 5 mod 7 = '0101'
    qc.add_classical(3, y_reg)
    qc.add_classical_modulo(X=2, y_reg=y_reg, ancilla_bit=anc_reg[0], N=7)
    qc.measure(y_reg, output_reg)
    qc.measure(anc_reg, anc_ouput_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    a_dist = res.data.anc_output.get_counts()

    assert dist["0101"] == 1
    assert a_dist["0"] == 1

    # Without ancilla clean up
    qc = AdderCircuit(y_reg, output_reg, anc_reg, anc_ouput_reg)
    # 0 + 3 + 2 = 5 mod 7 = '0101'
    qc.add_classical(3, y_reg)
    qc.add_classical_modulo(X=2, y_reg=y_reg, ancilla_bit=anc_reg[0], N=7, clean_up_ancilla=False)
    qc.measure(y_reg, output_reg)
    qc.measure(anc_reg, anc_ouput_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    a_dist = res.data.anc_output.get_counts()

    assert dist["0101"] == 1
    assert a_dist["1"] == 1


def test_c_add_classical_modulo() -> None:
    y_reg = QuantumRegister(4)
    output_reg = ClassicalRegister(4, name="output")
    c_reg = QuantumRegister(1)
    anc_reg = QuantumRegister(1)
    anc_ouput_reg = ClassicalRegister(1, name="anc_output")

    # Perform modulo 7 operations.
    N = 7

    # Control bit = |0>
    qc = AdderCircuit(c_reg, y_reg, output_reg, anc_reg, anc_ouput_reg)
    # Add 3 to y register
    qc.add_classical(3, y_reg)
    # Control_add 2 to y register
    qc.c_add_classical_modulo(c_reg, 2, y_reg, anc_reg[0], N)
    qc.measure(y_reg, output_reg)
    qc.measure(anc_reg, anc_ouput_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    a_dist = res.data.anc_output.get_counts()

    assert dist["0011"] == 1
    assert a_dist["0"] == 1

    # Control bit = |1>
    qc = AdderCircuit(c_reg, y_reg, output_reg, anc_reg, anc_ouput_reg)
    qc.x(c_reg[0])
    # Add 3 to y register
    qc.add_classical(3, y_reg)
    # Control_add 2 to y register
    qc.c_add_classical_modulo(c_reg, 2, y_reg, anc_reg[0], N)
    qc.measure(y_reg, output_reg)
    qc.measure(anc_reg, anc_ouput_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    a_dist = res.data.anc_output.get_counts()

    assert dist["0101"] == 1
    assert a_dist["0"] == 1


def test_add_quantum() -> None:
    x_reg = QuantumRegister(3)
    y_reg = QuantumRegister(4)
    outpout_reg = ClassicalRegister(4, name="output")
    qc = AdderCircuit(x_reg, y_reg, outpout_reg)

    # Add 3 in x register
    qc.add_classical(3, x_reg)
    # Add 6 in y register
    qc.add_classical(6, y_reg)
    # Add 10 times x register to y register
    qc.add_quantum(x_reg, y_reg, A=10)
    qc.measure(y_reg, outpout_reg)

    # Expected result = 10*3 + 6 = 4 mod 16 = "0100"
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["0100"] == 1


def test_c_add_quantum() -> None:
    x_reg = QuantumRegister(3)
    y_reg = QuantumRegister(4)
    outpout_reg = ClassicalRegister(4, name="output")
    c_reg = QuantumRegister(1)

    # Control bit = |0>
    qc = AdderCircuit(c_reg, x_reg, y_reg, outpout_reg)
    # Add 3 in x register
    qc.add_classical(3, x_reg)
    # Add 6 in y register
    qc.add_classical(6, y_reg)
    # Add 10 times x register to y register
    qc.c_add_quantum(c_reg, x_reg, y_reg, A=10)
    qc.measure(y_reg, outpout_reg)
    # Expected result = 6 mod 16 = "0110"
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["0110"] == 1

    # Control bit = |1>
    qc = AdderCircuit(c_reg, x_reg, y_reg, outpout_reg)
    qc.x(c_reg[0])
    # Add 3 in x register
    qc.add_classical(3, x_reg)
    # Add 6 in y register
    qc.add_classical(6, y_reg)
    # Add 10 times x register to y register
    qc.c_add_quantum(c_reg, x_reg, y_reg, A=10)
    qc.measure(y_reg, outpout_reg)

    # Expected result = 10*3 + 6 = 4 mod 16 = "0100"
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["0100"] == 1


def test_add_quantum_modulo() -> None:
    x_reg = QuantumRegister(3)
    y_reg = QuantumRegister(4)
    outpout_reg = ClassicalRegister(4, name="output")
    anc_reg = QuantumRegister(1)
    anc_ouput_reg = ClassicalRegister(1, name="anc_output")
    qc = AdderCircuit(x_reg, y_reg, outpout_reg, anc_reg, anc_ouput_reg)

    # Perform modulo 7 operations.
    N = 7
    # Add 4 in x register
    qc.add_classical(4, x_reg)
    # Add 6 in y register
    qc.add_classical(6, y_reg)
    # Add 10 times x register to y register modulo 7
    qc.add_quantum_modulo(x_reg, y_reg, anc_reg[0], N, A=10)
    qc.measure(y_reg, outpout_reg)
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
    y_reg = QuantumRegister(4)
    outpout_reg = ClassicalRegister(4, name="output")
    anc_reg = QuantumRegister(1)
    anc_ouput_reg = ClassicalRegister(1, name="anc_output")

    # Perform modulo 7 operations.
    N = 7

    # Case 1: Control bit = |0>
    qc = AdderCircuit(control_reg, x_reg, y_reg, outpout_reg, anc_reg, anc_ouput_reg)
    # Add 4 in x register
    qc.add_classical(4, x_reg)
    # Add 6 in y register
    qc.add_classical(6, y_reg)
    # Add 10 times x register to y register modulo 7
    qc.c_add_quantum_modulo(control_reg, x_reg, y_reg, anc_reg[0], N, A=10)
    qc.measure(y_reg, outpout_reg)
    qc.measure(anc_reg, anc_ouput_reg)
    # Expected result = 6 mod 7 = "0110"
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    a_dist = res.data.anc_output.get_counts()
    assert dist["0110"] == 1
    assert a_dist["0"] == 1

    # Case 2: Control bit = |1>
    qc = AdderCircuit(control_reg, x_reg, y_reg, outpout_reg, anc_reg, anc_ouput_reg)
    qc.x(control_reg[0])
    # Add 4 in x register
    qc.add_classical(4, x_reg)
    # Add 6 in y register
    qc.add_classical(6, y_reg)
    # Add 10 times x register to y register modulo 7
    qc.c_add_quantum_modulo(control_reg, x_reg, y_reg, anc_reg[0], N, A=10)
    qc.measure(y_reg, outpout_reg)
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
    ancilla_reg = QuantumRegister(2)
    outpout_reg = ClassicalRegister(2 * n + 2, name="output")
    o_bit = ancilla_reg[0]
    a_bit = ancilla_reg[1]

    # Case 1: with uncomputation, with swap
    qc = AdderCircuit(x_reg, y_reg, outpout_reg, ancilla_reg)
    # Add 3 in x register
    qc.add_classical(3, x_reg)
    # |3> ->  |9*3 modulo 5>
    qc.multiply_modulo(A=9, x_reg=x_reg, y_reg=y_reg, overflow_bit=o_bit, ancilla_bit=a_bit, N=N)
    qc.measure(x_reg[:] + y_reg[:] + ancilla_reg[:], outpout_reg)
    # Expected x_reg value = 9*3 mod 5 = 2 = "010" -> Less significant output bits
    # Expected y_reg value = "000" -> Middle output bits
    # Expected ancilla = "00" -> More significant output bits
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["00000010"] == 1

    # Case 2: with uncomputation, without swap
    qc = AdderCircuit(x_reg, y_reg, outpout_reg, ancilla_reg)
    # Add 3 in x register
    qc.add_classical(3, x_reg)
    # |3> ->  |9*3 modulo 5>
    qc.multiply_modulo(
        A=9, x_reg=x_reg, y_reg=y_reg, overflow_bit=o_bit, ancilla_bit=a_bit, N=N, with_swap=False
    )
    qc.measure(x_reg[:] + y_reg[:] + ancilla_reg[:], outpout_reg)
    # Expected x_reg value = "000"
    # Expected y_reg value = 9*3 mod 5 = 2 = "010"
    # Expected ancilla = "00"
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["00010000"] == 1

    # Case 3: without uncomputation, with swap
    qc = AdderCircuit(x_reg, y_reg, outpout_reg, ancilla_reg)
    # Add 3 in x register
    qc.add_classical(3, x_reg)
    # |3> ->  |9*3 modulo 5>
    qc.multiply_modulo(
        A=9, x_reg=x_reg, y_reg=y_reg, overflow_bit=o_bit, ancilla_bit=a_bit, N=N, with_uncomputation=False
    )
    qc.measure(x_reg[:] + y_reg[:] + ancilla_reg[:], outpout_reg)
    # Expected x_reg value = 9*3 mod 5 = 2 = "010" -> Less significant output bits
    # Expected y_reg value = 3 ="011" -> Middle output bits
    # Expected ancilla = "00" -> More significant output bits
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["00011010"] == 1

    # Case 4: without uncomputation, without swap
    qc = AdderCircuit(x_reg, y_reg, outpout_reg, ancilla_reg)
    # Add 3 in x register
    qc.add_classical(3, x_reg)
    # |3> ->  |9*3 modulo 5>
    qc.multiply_modulo(
        A=9,
        x_reg=x_reg,
        y_reg=y_reg,
        overflow_bit=o_bit,
        ancilla_bit=a_bit,
        N=N,
        with_swap=False,
        with_uncomputation=False,
    )
    qc.measure(x_reg[:] + y_reg[:] + ancilla_reg[:], outpout_reg)
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
    ancilla_reg = QuantumRegister(2)
    outpout_reg = ClassicalRegister(2 * n + 2, name="output")

    o_bit = ancilla_reg[0]
    a_bit = ancilla_reg[1]

    # Case 1: control bits = |01> (no operation)
    qc = AdderCircuit(control_reg, x_reg, y_reg, outpout_reg, ancilla_reg)
    qc.x(control_reg[1])
    # Add 3 = "011" in x register
    qc.add_classical(3, x_reg)
    qc.c_multiply_modulo(
        control_reg=control_reg, A=9, x_reg=x_reg, y_reg=y_reg, overflow_bit=o_bit, ancilla_bit=a_bit, N=N
    )
    qc.measure(x_reg[:] + y_reg[:] + ancilla_reg[:], outpout_reg)
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    print(dist)
    assert dist["00000011"] == 1

    # Case 2: control bits = |11>
    qc = AdderCircuit(control_reg, x_reg, y_reg, outpout_reg, ancilla_reg)
    qc.x(control_reg[0])
    qc.x(control_reg[1])
    # Add 3 = "011" in x register
    qc.add_classical(3, x_reg)
    qc.c_multiply_modulo(
        control_reg=control_reg, A=9, x_reg=x_reg, y_reg=y_reg, overflow_bit=o_bit, ancilla_bit=a_bit, N=N
    )
    qc.measure(x_reg[:] + y_reg[:] + ancilla_reg[:], outpout_reg)
    # Expected x_reg value = 9*3 mod 5 = 2 = "010" -> Less significant output bits
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["00000010"] == 1


def test_exponentiate_modulo() -> None:
    # Perform modulo 5 operations.
    N = 5
    n = math.ceil(math.log2(N))  # = 3
    m = 6
    x_reg = QuantumRegister(m)
    y_reg = QuantumRegister(n)
    ancilla_reg = QuantumRegister(n + 2)
    outpout_reg = ClassicalRegister(m + 2 * n + 2, name="output")

    qc = AdderCircuit(x_reg, y_reg, ancilla_reg, outpout_reg)
    qc.add_classical(4, x_reg)
    qc.add_classical(3, y_reg)
    # |4> |3> |0> ->  |4> |2^4 * 3 modulo 5> |0>
    qc.exponentiate_modulo(A=2, x_reg=x_reg, y_reg=y_reg, ancilla_reg=ancilla_reg, N=N)
    qc.measure(x_reg[:] + y_reg[:] + ancilla_reg[:], outpout_reg)
    # Expected x_reg value = 4 = "000100"
    # Expected y_reg value = 2^4 * 3 mod 5 = 3 = "011"
    # Expected ancilla = "00000"
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["00000" + "011" + "000100"] == 1

    qc = AdderCircuit(x_reg, y_reg, ancilla_reg, outpout_reg)
    qc.add_classical(20, x_reg)
    qc.add_classical(1, y_reg)
    # |20> |1> |0> ->  |20> |2^20 * 1 modulo 5> |0>
    qc.exponentiate_modulo(A=2, x_reg=x_reg, y_reg=y_reg, ancilla_reg=ancilla_reg, N=N)
    qc.measure(x_reg[:] + y_reg[:] + ancilla_reg[:], outpout_reg)
    # Expected x_reg value = 20 = "010100"
    # Expected y_reg value = 2^20 * 1 modulo 5 = 1 = "001"
    # Expected ancilla = "00000"
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    print(dist)
    assert dist["00000" + "001" + "010100"] == 1


def test_add_quantum_modulo_RC() -> None:
    # Perform modulo 9 operations with y_init = 0.
    N = 9
    n = math.ceil(math.log2(N))  # = 4
    p = math.ceil(math.log2(n))  # = 2
    x_reg = QuantumRegister(n)
    y_reg = QuantumRegister(n + p)
    outpout_reg = ClassicalRegister(n + p, name="output")

    qc = AdderCircuit(x_reg, y_reg, outpout_reg)
    # Add 2 in x register
    qc.add_classical(2, x_reg)
    # Add 3 times x register to y register modulo 9
    qc.add_quantum_modulo_RC(x_reg, y_reg, N, a=3)
    qc.measure(y_reg, outpout_reg)
    # Expected result = 2*3 mod 9 = 6 = "110" -> Less significant output bits
    # Expected ancilla "000" -> More significant output bits
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["000110"] == 1

    qc = AdderCircuit(x_reg, y_reg, outpout_reg)
    # Add 7 in x register
    qc.add_classical(7, x_reg)
    # Add 8 times x register to y register modulo 9
    qc.add_quantum_modulo_RC(x_reg, y_reg, N, a=8)
    qc.measure(y_reg, outpout_reg)
    # Expected result = 7*8 mod 9 = 2 = "010" -> Less significant output bits
    # Expected ancilla "00" -> More significant output bits
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["000010"] == 1

    # Perform modulo 7 operations with y_init = 6.
    N = 7
    n = math.ceil(math.log2(N))  # = 3
    p = math.ceil(math.log2(n))  # = 2
    x_reg = QuantumRegister(n)
    y_reg = QuantumRegister(n + p)
    outpout_reg = ClassicalRegister(n + p, name="output")
    # Case 1: with uncomputation
    qc = AdderCircuit(x_reg, y_reg, outpout_reg)
    # Add 4 in x register
    qc.add_classical(4, x_reg)
    # Add 6 in y register
    qc.add_classical(6, y_reg)
    # Add 5 times x register to y register modulo 7
    qc.add_quantum_modulo_RC(x_reg, y_reg, N, a=5)
    qc.measure(y_reg, outpout_reg)

    # Expected result = (5*4 + 6) mod 7 = 5 = "101" -> Less significant output bits
    # Expected ancilla = 6 mod 4 = 2 "10" -> More significant output bits
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["10101"] == 1

    # Case 2: without uncomputation
    qc = AdderCircuit(x_reg, y_reg, outpout_reg)
    # Add 4 in x register
    qc.add_classical(4, x_reg)
    # Add 6 in y register
    qc.add_classical(6, y_reg)
    # Add 5 times x register to y register modulo 7
    qc.add_quantum_modulo_RC(x_reg, y_reg, N, a=5, with_uncomputation=False)
    qc.measure(y_reg, outpout_reg)

    # Expected result = (5*4 + 6) mod 7 = 5 = "101" -> Less significant output bits
    # Expected ancilla = ((5*4 mod 7) + 6) // 7 = 1 = "01" -> More significant output bits
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["01101"] == 1


def test_c_add_quantum_modulo_RC() -> None:
    # Perform modulo 9 operations with y_init = 0.
    N = 9
    n = math.ceil(math.log2(N))  # = 4
    p = math.ceil(math.log2(n))  # = 2
    control_reg = QuantumRegister(1)
    x_reg = QuantumRegister(n)
    y_reg = QuantumRegister(n + p)
    outpout_reg = ClassicalRegister(n + p, name="output")

    # Case 1: control bit = |0>
    qc = AdderCircuit(control_reg, x_reg, y_reg, outpout_reg)
    # Add 4 in x register
    qc.add_classical(4, x_reg)
    # Add 6 in y register
    qc.add_classical(6, y_reg)
    # Add 10 times x register to y register modulo 9
    qc.c_add_quantum_modulo_RC(control_reg, x_reg, y_reg, N, a=10)
    qc.measure(y_reg, outpout_reg)

    # Expected result 6 mod 9 = "0110" -> Less significant output bits
    # Expected ancilla "00" -> More significant output bits
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["000110"] == 1

    # Case 2: control bit = |1>
    qc = AdderCircuit(control_reg, x_reg, y_reg, outpout_reg)
    qc.x(control_reg[0])
    # Add 4 in x register
    qc.add_classical(4, x_reg)
    # Add 6 in y register
    qc.add_classical(6, y_reg)
    # Add 10 times x register to y register modulo 9
    qc.c_add_quantum_modulo_RC(control_reg, x_reg, y_reg, N, a=10)
    qc.measure(y_reg, outpout_reg)

    # Expected result: 10*4 + 6 mod 9 = 1 = "0001" -> Less significant output bits
    # Expected ancilla: 6 mod 4 = 2 = "10" -> More significant output bits
    res = run_simulation(qc)
    dist = res.data.output.get_counts()
    assert dist["100001"] == 1
