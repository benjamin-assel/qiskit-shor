import math
from abc import ABC, abstractmethod

from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.library import SwapGate


class AdderCircuit(ABC, QuantumCircuit):
    """
    Quantum circuit implementing modular arithmetics used in the order-finding circuit of Shor's algorithm.
    Based on https://arxiv.org/abs/quant-ph/0205095 by Beauregard.

    Conventions:
    - classical integer X (capital letter): a Python integer.
    - quantum integer x (small letter): quantum state |x> =|x_0>|x_1> ... |x_m>, with x_k in {0, 1},
    where x = sum_k x_k 2^k. It corresponds to the binary string "x_m ... x_1 x_0" and it is ordered
    in the qubit register as [qubit_0, qubit_1, ..., qubit_m], where qubit_k is in the state |x_k>.
    Often the quantum register in the state representing x is called "x_reg".
    """

    @staticmethod
    def get_qubits(reg: QuantumRegister | list[Qubit] | Qubit) -> list[Qubit]:
        """Returns the provided qubits input as a list of qubits."""
        if isinstance(reg, Qubit):
            return [reg]
        return reg[:]

    @abstractmethod
    def add_classical(
        self,
        X: int,
        y_reg: QuantumRegister | list[Qubit],
        a_reg: QuantumRegister | list[Qubit] | None = None,
    ) -> None:
        """
        Adds the classical integer X to the quantum integer y, using optional ancilla qubits.

        Operation: |y> -> |X + y >

        The operation is performed modulo 2^n, where n is the size of the y register.
        The ancilla qubits are reset to zero after the operation.

        Arguments:
        - X: integer.
        - y_reg: quantum register or list of qubits.
        - a_reg: [Optional] quantum register or list of ancilla qubits.
        """
        pass

    @abstractmethod
    def c_add_classical(
        self,
        control_reg: QuantumRegister | list[Qubit] | Qubit,
        X: int,
        y_reg: QuantumRegister | list[Qubit],
        a_reg: QuantumRegister | list[Qubit] | None = None,
    ) -> None:
        """
        Controlled version of 'add_classical'.
        """
        pass

    def add_classical_modulo(
        self,
        X: int,
        y_reg: QuantumRegister | list[Qubit],
        ancilla_bit: Qubit,
        overflow_bit: Qubit,
        N: int,
        a_reg: QuantumRegister | list[Qubit] | None = None,
        reset_ancilla: bool = True,
    ) -> None:
        """
        Adds the classical integer X to the integer y modulo N, using
        one overflow qubit and one ancilla qubit.
        An optional ancilla register is used to perform additions.

        Operation:
            |y>|0> -> |r>|0>  if reset_ancilla = True
                   -> |r>|q>  else
            where r = (X + y) mod N, X + y = qN + r

        Assumptions:
            - 0 <= X < N
            - 0 <= y < N
            - y_reg has n qubits, where n = ceil(log2(N))
            - The ancilla_bit and overflow qubits are in the |0> state.
            - The optional ancilla register a_reg is in the |0> state (it is reset to |0> after the operation).
        """
        y_bits = y_reg[:] + [overflow_bit]
        # Number of bits to hold modulo N results.
        n = math.ceil(math.log2(N))

        assert 0 <= X and X < N, "X must be smaller than N."
        assert len(y_bits) == n + 1, f"The y register must have {n} qubits."

        self.add_classical(X - N, y_bits, a_reg)
        self.cx(y_bits[n], ancilla_bit)
        self.c_add_classical(ancilla_bit, N, y_bits, a_reg)

        if reset_ancilla:
            self.add_classical(-X, y_bits, a_reg)
            self.cx(y_bits[n], ancilla_bit)
            self.x(ancilla_bit)
            self.add_classical(X, y_bits, a_reg)

    def c_add_classical_modulo(
        self,
        control_reg: QuantumRegister | list[Qubit] | Qubit,
        X: int,
        y_reg: QuantumRegister | list[Qubit],
        ancilla_bit: Qubit,
        overflow_bit: Qubit,
        N: int,
        a_reg: QuantumRegister | list[Qubit] | None = None,
        reset_ancilla: bool = True,
    ) -> None:
        """
        Controlled version of 'add_classical_modulo'.
        """
        control_bits = self.get_qubits(control_reg)
        y_bits = y_reg[:] + [overflow_bit]
        # Number of bits to hold modulo N results.
        n = math.ceil(math.log2(N))

        assert 0 <= X and X < N, "X must be smaller than N."
        assert len(y_bits) == n + 1, f"The y register must have {n} qubits."

        self.c_add_classical(control_bits, X, y_bits, a_reg)
        self.add_classical(-N, y_bits, a_reg)
        self.cx(y_bits[n], ancilla_bit)
        self.c_add_classical(ancilla_bit, N, y_bits, a_reg)

        if reset_ancilla:
            self.add_classical(-X, y_bits, a_reg)
            self.mcx(control_bits + [y_bits[n]], ancilla_bit)
            self.x(ancilla_bit)
            self.add_classical(X, y_bits, a_reg)

    def add_quantum_modulo(
        self,
        x_reg: QuantumRegister | list[Qubit],
        y_reg: QuantumRegister | list[Qubit],
        ancilla_bit: Qubit,
        overflow_bit: Qubit,
        N: int,
        a_reg: QuantumRegister | list[Qubit] | None = None,
        A: int = 1,
    ) -> None:
        """
        Adds A times x to the quantum integer y modulo N, where A is a classical integer
        and x is a quantum integer, using one ancilla qubit.
        An optional ancilla register is used to perform additions.

        Operation:
            |x>|y>|0> -> |x>|(y + Ax) mod N >|0>

        Assumptions:
            - 0 <= x < N,
            - 0 <= y < N,
            - y_reg has n+1 qubits, where n = ceil(log2(N))
            - the ancilla_bit qubit is in the |0> state.
        """
        x_bits = x_reg[:]
        y_bits = y_reg[:]
        # Number of bits to hold modulo N results.
        n = math.ceil(math.log2(N))
        m = len(x_bits)

        assert m <= n, "x register may hold too large numbers."
        assert len(y_bits) == n, f"The y register must have {n} qubits."

        for i in range(m):
            self.c_add_classical_modulo(
                control_reg=x_bits[i],
                X=((A % N) * 2**i) % N,
                y_reg=y_bits,
                ancilla_bit=ancilla_bit,
                overflow_bit=overflow_bit,
                N=N,
                a_reg=a_reg,
            )

    def c_add_quantum_modulo(
        self,
        control_reg: QuantumRegister | list[Qubit] | Qubit,
        x_reg: QuantumRegister | list[Qubit],
        y_reg: QuantumRegister | list[Qubit],
        ancilla_bit: Qubit,
        overflow_bit: Qubit,
        N: int,
        a_reg: QuantumRegister | list[Qubit] | None = None,
        A: int = 1,
    ) -> None:
        """
        Controlled version of 'add_quantum_modulo'.
        """
        control_bits = self.get_qubits(control_reg)
        x_bits = x_reg[:]
        y_bits = y_reg[:]
        # Number of bits to hold modulo N results.
        n = math.ceil(math.log2(N))
        m = len(x_bits)

        assert m <= n, "x register may hold too large numbers."
        assert len(y_bits) == n, f"The y register must have {n} qubits."

        for i in range(m):
            self.c_add_classical_modulo(
                control_reg=control_bits + [x_bits[i]],
                X=((A % N) * 2**i) % N,
                y_reg=y_bits,
                ancilla_bit=ancilla_bit,
                overflow_bit=overflow_bit,
                N=N,
                a_reg=a_reg,
            )

    def multiply_modulo(
        self,
        A: int,
        x_reg: QuantumRegister | list[Qubit],
        y_reg: QuantumRegister | list[Qubit],
        overflow_bit: Qubit,
        ancilla_bit: Qubit,
        N: int,
        a_reg: QuantumRegister | list[Qubit] | None = None,
        with_uncomputation: bool = True,
        with_swap: bool = True,
    ) -> None:
        """
        Performs in-place multiplication x ->  Ax mod N, leveraging out-of-place
        addition on the y_reg qubits, where A is a classical integer.
        The computation requires two more ancillas: the "overflow_bit" and the "ancilla_bit".
        An optional ancilla register is used to perform additions.

        Operation:
            |x>_n |0>_n |0>|0> -> |Ax mod N >_n |0>_n |0>|0>

        Assumptions:
            - 0 <= x < N,
            - x_reg and y_reg to have exactly n = ceil(log2(N)) qubits
            - the y_reg qubits, overflow_bit and ancilla_bit are in the |0> state.

        If with_uncomputation=False, performs instead the operation: |x>|0>|0> -> |Ax mod N>|x>|0>.
        If with_swap=False, performs instead the operation: |x>|0>|0> -> |0>|Ax mod N >|0>.
        If both options are False, performs the operation: |x>|0>|0> -> |x>|Ax mod N>|0>.
        """
        x_bits = x_reg[:]
        y_bits = y_reg[:]
        n = math.ceil(math.log2(N))

        assert n == len(x_bits), f"The x register must have {n} qubits."
        assert n == len(y_bits), f"The y register must have {n} qubits."

        # Out-of-place a-multiplication stage: |x>|0>|0> ->|x>|ax mod N>|0>
        self.add_quantum_modulo(
            x_reg=x_bits,
            y_reg=y_bits,
            ancilla_bit=ancilla_bit,
            overflow_bit=overflow_bit,
            N=N,
            a_reg=a_reg,
            A=A,
        )
        if with_swap:
            # Swap stage : |x>|ax mod N>|0> -> |ax mod N>|x>|0>
            for i in range(n):
                self.swap(x_bits[i], y_bits[i])
        if with_uncomputation:
            # Uncomputation stage: |ax mod N>|x>|0> -> |ax mod N>|0>|0>
            B = pow(A, -1, N)  # AB = 1 mod N
            x = x_bits if with_swap else y_bits
            y = y_bits if with_swap else x_bits
            self.add_quantum_modulo(
                x_reg=x, y_reg=y, ancilla_bit=ancilla_bit, overflow_bit=overflow_bit, N=N, a_reg=a_reg, A=-B
            )

    def c_multiply_modulo(
        self,
        control_reg: QuantumRegister | list[Qubit] | Qubit,
        A: int,
        x_reg: QuantumRegister | list[Qubit],
        y_reg: QuantumRegister | list[Qubit],
        overflow_bit: Qubit,
        ancilla_bit: Qubit,
        N: int,
        a_reg: QuantumRegister | list[Qubit] | None = None,
        with_uncomputation: bool = True,
        with_swap: bool = True,
    ) -> None:
        """
        Controlled version of 'multiply_modulo'.
        """
        control_bits = self.get_qubits(control_reg)
        x_bits = x_reg[:]
        y_bits = y_reg[:]
        k = len(control_bits)
        n = math.ceil(math.log2(N))

        assert n == len(x_bits), f"The x register must have {n} qubits."
        assert n == len(y_bits), f"The y register must have {n} qubits."

        # Out-of-place a-multiplication stage
        self.c_add_quantum_modulo(
            control_reg=control_bits,
            x_reg=x_bits,
            y_reg=y_bits,
            ancilla_bit=ancilla_bit,
            overflow_bit=overflow_bit,
            N=N,
            a_reg=a_reg,
            A=A,
        )
        if with_swap:
            # Swap stage
            for i in range(n):
                cswap_gate = SwapGate().control(k)
                self.append(cswap_gate, control_bits + [x_bits[i], y_bits[i]])
        if with_uncomputation:
            # Uncomputation stage
            B = pow(A, -1, N)  # AB = 1 mod N
            x = x_bits if with_swap else y_bits
            y = y_bits if with_swap else x_bits
            self.c_add_quantum_modulo(
                control_reg=control_bits,
                x_reg=x,
                y_reg=y,
                ancilla_bit=ancilla_bit,
                overflow_bit=overflow_bit,
                N=N,
                a_reg=a_reg,
                A=-B,
            )

    def exponentiate_modulo(
        self,
        A: int,
        x_reg: QuantumRegister | list[Qubit],
        y_reg: QuantumRegister | list[Qubit],
        ancilla_reg: QuantumRegister | list[Qubit],
        N: int,
        a_reg: QuantumRegister | list[Qubit] | None = None,
    ) -> None:
        """
        Performs modulo N multiplication of y by A^x, using n+2 ancilla qubits,
        where A is a classical integer.
        An optional ancilla register is used to perform additions.

        Operation: |x>_m |y>_n |0>_{n+2} -> |x>_m |(A^x * y) mod N >_n |0>_{n+2}

        Assumptions:
            - 0 <= y < N,
            - size(y_reg) == n,
            - size(ancilla_reg) == n+2,
            where n = ceil(log2(N)).
        """
        x_bits = x_reg[:]
        y_bits = y_reg[:]
        a_bits = ancilla_reg[:]
        n = math.ceil(math.log2(N))
        m = len(x_bits)

        assert len(y_bits) == n, f"The y register must have {n} qubits."
        assert len(ancilla_reg) == n + 2, f"The ancilla register must have {n + 2} qubits."

        for i in range(m):
            self.c_multiply_modulo(
                control_reg=x_bits[i],
                A=pow(A, 2**i, N),
                x_reg=y_bits,
                y_reg=a_bits[:n],
                overflow_bit=a_bits[n],
                ancilla_bit=a_bits[n + 1],
                N=N,
                a_reg=a_reg,
            )
