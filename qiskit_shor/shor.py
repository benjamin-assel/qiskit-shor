import math
import random
from fractions import Fraction

from qiskit.circuit import ClassicalRegister, QuantumRegister
from qiskit.circuit.library import QFTGate

from qiskit_shor.adder import AdderCircuit


def order_finding_circuit(A: int, N: int, precision: int = 0) -> AdderCircuit:
    """
    Build circuit to find the order of A in Z_N.
    Args:
        A: int.
        N: int.
        precision: Number of qubits to use for phase estimation. Default value: 2*ceil(log2(N)).
    Returns:
        AdderCircuit: Order finding circuit.
    """
    if math.gcd(A, N) > 1:
        print(f"Error: gcd({A},{N}) > 1")
        return 0

    n = math.ceil(math.log2(N))
    m = precision if precision > 0 else 2 * n

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


def find_order(
    A: int,
    N: int,
    sampler,
    pass_manager,
    precision: int = 0,
    max_tries: int = 10,
) -> int:
    """
    Carry out search algorithm for fnding the order of the integer A in Z_N, i.e. the
    integer r such that A^r = 1 mod N, on a simulator.
    Assumes that N is odd, N is not a power of a prime integer and A and N are coprime.
    Args:
        A: int.
        N: int.
        precision: Number of qubits to use for phase estimation. Default value: 2*ceil(log2(N)).
        max_tries: Number of trials (circuit sampling runs).
    Returns:
        int: Found order or zero if no success.
    """
    qc = order_finding_circuit(A, N, precision)
    m = qc.cregs[0].size
    qc_isa = pass_manager.run(qc)

    print(f"Start search for the order of {A} in Z_{N}")
    i = 0
    while i < max_tries:
        dist = sampler.run([qc_isa], shots=1).result()[0].data.output_bits.get_counts()
        x = int(max(dist, key=dist.get), 2)
        r = Fraction(x / 2**m).limit_denominator(N).denominator
        if pow(A, r, N) == 1:
            print(f"Found order of {A} in Z_{N}: {r}")
            return r
        print(f"\t Trial {i}: incorrect value {r}")
        i += 1
    print(f"Failed to find order of {A} in Z_{N}")
    return 0


def find_factor(
    N: int, sampler, pm, num_tries: int = 3, order_finding_max_tries: int = 3, seed: int | None = None
) -> int:
    """
    Carry out search algorithm for finding a factor of N.
    Args:
        N: int.
        sampler: Sampler.
        pm: Pass manager.
        num_tries: Number of trials.
        order_finding_max_tries: Number of trials (circuit sampling runs) for order finding.
        seed: Random seed.
    Returns:
        int: Found factor or zero if no success.
    """
    # Check if N is even or a non-trivial power.
    if N % 2 == 0:
        print("Even number")
        return 2

    for k in range(2, round(math.log(N, 2)) + 1):
        d = int(round(N ** (1 / k)))
        if d**k == N:
            factor_found = True
            print(f"Number is {d} to the power {k}")
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
        r = find_order(a, N, sampler, pm, max_tries=order_finding_max_tries)
        if r == 0:
            continue
        if r % 2 == 0:
            x = pow(a, r // 2, N) - 1
            d = math.gcd(x, N)
            if d > 1:
                factor_found = True
        i += 1

    if factor_found:
        print(f"Factor found: {d}")
        return d

    print("No factor found")
    return 0
