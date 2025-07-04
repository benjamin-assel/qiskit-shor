import math
import random
from fractions import Fraction

from qiskit.circuit import ClassicalRegister, QuantumRegister
from qiskit.circuit.library import QFTGate

from qiskit_shor.adder import AdderCircuit


def order_finding_circuit(A: int, N: int, precision: int | None = None) -> AdderCircuit:
    """
    Build circuit to find the order of A in Z_N, using 4n+2 qubits, with n = ceil(log2(N)).
    Args:
        A: int.
        N: int.
        precision: Number of qubits to use for phase estimation. If None, use default value: 2n.
    Returns:
        AdderCircuit: Order finding circuit.
    """
    if math.gcd(A, N) > 1:
        print(f"Error: gcd({A},{N}) > 1")
        return 0

    n = math.ceil(math.log2(N))
    m = precision if precision is not None else 2 * n

    control_register = QuantumRegister(m)
    target_register = QuantumRegister(n)
    ancilla_register = QuantumRegister(n + 2)
    output_register = ClassicalRegister(m, name="output_bits")
    qc = AdderCircuit(control_register, target_register, ancilla_register, output_register)

    # Prepare control state in "all quantum integers" superposition state
    for i in range(m):
        qc.h(control_register[i])

    # Prepare target state in |1> state
    qc.x(target_register[0])

    # Apply modular exponential operator
    qc.exponentiate_modulo(
        A=A, x_reg=control_register, y_reg=target_register, ancilla_reg=ancilla_register, N=N
    )

    # Apply inverse QFT
    qc.compose(QFTGate(m).inverse(), qubits=control_register, inplace=True)

    # Measure control state
    qc.measure(control_register, output_register)

    return qc


def order_finding_circuit_one_control(A: int, N: int, precision: int | None = None) -> AdderCircuit:
    """
    Build circuit to find the order of A in Z_N, using modular multiplication with a single control qubit
    and repeated measurements. The circuit uses 2n + 3 qubits in total, with n = ceil(log2(N)).
    Args:
        A: int.
        N: int.
        precision: int. Number of qubits to use for phase estimation. If None, use default value: 2n.
    Returns:
        AdderCircuit: Order finding circuit.
    """
    if math.gcd(A, N) > 1:
        print(f"Error: gcd({A},{N}) > 1")
        return 0

    n = math.ceil(math.log2(N))
    m = precision if precision is not None else 2 * n

    control_register = QuantumRegister(1)
    target_register = QuantumRegister(n)
    ancilla_register = QuantumRegister(n + 2)
    output_register = ClassicalRegister(m, name="output_bits")
    qc = AdderCircuit(control_register, target_register, ancilla_register, output_register)

    # Prepare target state in |1> state
    qc.x(target_register[0])

    # Sequential measurements
    c_bit = control_register[0]
    for i in range(m):
        # Control bit preparation
        qc.h(c_bit)
        # Controlled modular multiplication
        qc.c_multiply_modulo(
            control_reg=c_bit,
            A=pow(A, 2 ** (m - i - 1), N),
            x_reg=target_register,
            y_reg=ancilla_register[:n],
            overflow_bit=ancilla_register[n],
            ancilla_bit=ancilla_register[n + 1],
            N=N,
        )
        # Inverse QFT i-th component.
        for j in range(i):
            with qc.if_test((output_register[j], 1)):
                qc.p(-math.pi / 2 ** (i - j), c_bit)
        qc.h(c_bit)
        # Measurement
        qc.measure(c_bit, output_register[i])
        # Reset
        with qc.if_test((output_register[i], 1)):
            qc.x(c_bit)
    return qc


def _get_order_from_dist(dist: dict, A: int, N: int, precision: int) -> int:
    sorted_outputs = sorted(dist, key=dist.get, reverse=True)
    # Look for the order in the 10 most frequent ouputs.
    for i in range(min(10, len(sorted_outputs))):
        if sorted_outputs[i] == "0" * precision:
            continue
        x = int(sorted_outputs[i], 2)
        r = Fraction(x / 2**precision).limit_denominator(N - 1).denominator
        if pow(A, r, N) == 1:
            print(
                f"""Found value {r} for the order of {A} in Z_{N}. If running on noisy quantum hardware, {r} might be a multiple of the order instead."""
            )
            return r
    print(f"Failed to find order of {A} in Z_{N}")
    return 0


def find_order(
    A: int,
    N: int,
    sampler,
    pass_manager,
    precision: int | None = None,
    num_shots: int = 10,
    one_control_circuit: bool = False,
) -> tuple[int, dict[str, int]]:
    """
    Carry out search algorithm for fnding the order of the integer A in Z_N, i.e. the
    integer r such that A^r = 1 mod N, on a simulator.
    Assumes that N is odd, N is not a power of a prime integer and A and N are coprime.
    Args:
        A: int.
        N: int.
        precision: Number of qubits to use for phase estimation. If None, use default value: 2*ceil(log2(N)).
        num_shots: Number of circuit sampling runs. Default value: 10.
        one_control_circuit: boolean. Use order finding circuit with a single control qubit. Default value: False.
    Returns:
        tuple[int, dict[str, int]]: The first element is the order (if found) or zero (if not). The second
                                    element is the distribution of the measurement outcomes.
    """
    m = precision if precision is not None else 2 * math.ceil(math.log2(N))
    if one_control_circuit:
        qc = order_finding_circuit_one_control(A, N, precision=m)
    else:
        qc = order_finding_circuit(A, N, precision=m)
    qc_isa = pass_manager.run(qc)

    print(f"Start search for the order of {A} in Z_{N}")
    dist = sampler.run([qc_isa], shots=num_shots).result()[0].data.output_bits.get_counts()
    r = _get_order_from_dist(dist, A, N, precision=m)
    return r, dist


def find_factor(
    N: int,
    sampler,
    pass_manager,
    num_tries: int = 3,
    num_shots_per_trial: int = 10,
    one_control_circuit: bool = False,
    seed: int | None = None,
) -> int:
    """
    Carry out search algorithm for finding a factor of N.
    Args:
        N: int.
        sampler: Sampler.
        pass_manager: Pass manager.
        num_tries: Number of trials.
        num_shots_per_trial: Number of order finding circuit runs per trial.
        one_control_circuit: boolean. Use order finding circuit with a single control qubit. Default value: False.
        seed: Random seed.
    Returns:
        int: Found factor or one if no success.
    """
    # Check if N is even or a non-trivial power.
    if N % 2 == 0:
        print("Even number")
        return 2

    for k in range(2, round(math.log(N, 2)) + 1):
        d = int(round(N ** (1 / k)))
        if d**k == N:
            factor_found = True
            print(f"{N} is {d} to the power {k}")
            return d

    i = 0
    factor_found = False
    if seed is not None:
        random.seed(seed)
    while (not factor_found) and i < num_tries:
        a = random.randint(2, N - 1)
        d = math.gcd(a, N)
        if d > 1:
            factor_found = True
            print(f"Lucky guess of {a}, found factor {d}")
            return d
        # Run order finding circuit
        r, _ = find_order(
            a,
            N,
            sampler,
            pass_manager,
            num_shots=num_shots_per_trial,
            one_control_circuit=one_control_circuit,
        )
        if r == 0:
            continue
        if r % 2 == 0:
            x = pow(a, r // 2, N) - 1
            d = math.gcd(x, N)
            if d > 1 and d < N:
                factor_found = True
        i += 1

    if factor_found:
        print(f"Factor found: {d}")
        return d

    print("No factor found")
    return 1
