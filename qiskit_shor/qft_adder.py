import math

import numpy as np
from qiskit.circuit import QuantumRegister, Qubit

from qiskit_shor.adder import AdderCircuit
from qiskit_shor.gate_utils import QFTFullGate


class QFTAdderCircuit(AdderCircuit):
    """
    Quantum circuit implementing modular arithmetics used in the order-finding circuit of Shor's algorithm.
    Operations are constructed from basic additions using Quantum Fourier Transform methods, as in
    https://arxiv.org/abs/quant-ph/0205095 by Beauregard.
    This method does not require ancilla qubits for (non-modular) additions.
    """

    # Whether to use approximate QFT gates in additions.
    approx_QFT: bool

    def __init__(self, *args, approx_QFT: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.approx_QFT = approx_QFT

    def qft_approx_degree(self, n: int):
        """The approximation degree to use in QFT gates (zero if no approximation is used).
        It is defined to drop phase gates with angle smaller than π/2^d, with d = ceil(log2(n)) + 2.
        See https://arxiv.org/pdf/quant-ph/0403071 for details."""
        return max(0, n - math.ceil(np.log2(n)) - 2) if self.approx_QFT else 0

    def add_classical(
        self,
        X: int,
        y_reg: QuantumRegister | list[Qubit],
        a_reg: QuantumRegister | list[Qubit] | None = None,
        include_QFT: bool = True,
    ) -> None:
        """
        Adds the classical integer X to the quantum integer y.

        Operation: |y> -> |X + y >

        The operation is performed modulo 2^n, where n is the size of the y register.

        Arguments:
        - X: integer.
        - y_reg: quantum register or list of qubits.
        - include_QFT: whether to include the QFT gates in the circuit.

        Gate counts: QFT gates -> O(n^2) (O(n log(n)) with approximation), other gates -> O(n).
        """
        assert a_reg is None, "QFTAdder additions do not need ancillas for additions."
        y_bits = self.get_qubits(y_reg)
        n = len(y_bits)

        if include_QFT:
            # QFT
            qft_gate = QFTFullGate(n, approximation_degree=self.qft_approx_degree(n))
            self.compose(qft_gate, y_bits, inplace=True)
        # Phase gates (= addition in Fourier space)
        for i in range(n):
            self.p(2 * np.pi * X * (2 ** (i - n)), y_bits[i])

        if include_QFT:
            # Inverse QFT
            inv_qft_gate = QFTFullGate(n, approximation_degree=self.qft_approx_degree(n)).inverse()
            self.compose(inv_qft_gate, y_bits, inplace=True)

    def c_add_classical(
        self,
        control_reg: QuantumRegister | list[Qubit] | Qubit,
        X: int,
        y_reg: QuantumRegister | list[Qubit],
        a_reg: QuantumRegister | list[Qubit] | None = None,
        include_QFT: bool = True,
    ) -> None:
        """
        Controlled version of 'add_classical'.
        """
        assert a_reg is None, "QFTAdder additions do not need ancillas for additions."
        control_bits = self.get_qubits(control_reg)
        y_bits = y_reg[:]
        n = len(y_bits)

        if include_QFT:
            # QFT
            qft_gate = QFTFullGate(n, approximation_degree=self.qft_approx_degree(n))
            self.compose(qft_gate, y_bits, inplace=True)
        # Controlled phase gates (= c_addition in Fourier space)
        for i in range(n):
            theta = 2 * np.pi * X * (2 ** (i - n))
            self.mcp(theta, control_bits, y_bits[i])

        if include_QFT:
            # Inverse QFT
            inv_qft_gate = QFTFullGate(n, approximation_degree=self.qft_approx_degree(n)).inverse()
            self.compose(inv_qft_gate, y_bits, inplace=True)

    def add_quantum(
        self,
        x_reg: QuantumRegister | list[Qubit],
        y_reg: QuantumRegister | list[Qubit],
        A: int = 1,
        include_QFT: bool = True,
    ) -> None:
        """
        Adds A times x to the quantum integer y, where A is a classical integer and x is a quantum integer.

        Operation:
            |x>|y> -> |x>|y + Ax>

        The operation is performed modulo 2^n, where n is the size of the y register.

        Arguments:
        - x_reg: quantum register or list of qubits.
        - y_reg: quantum register or list of qubits.
        - A: integer.
        - include_QFT: whether to include the QFT gates in the circuit.

        Gate counts: QFT gates -> O(n^2) (O(n log(n)) with approximation), other gates -> O(n^2).
        """
        x_bits = x_reg[:]
        y_bits = y_reg[:]
        m = len(x_bits)
        n = len(y_bits)

        if include_QFT:
            # QFT
            qft_gate = QFTFullGate(n, approximation_degree=self.qft_approx_degree(n))
            self.compose(qft_gate, y_bits, inplace=True)
        # Phase gates (= addition in Fourier space)
        for i in range(m):
            for j in range(n):
                if i + j < n:  # The phase rotation is trivial if i+j>=n.
                    self.cp(2 * np.pi * A * (2 ** (i + j - n)), x_bits[i], y_bits[j])
        if include_QFT:
            # Inverse QFT
            inv_qft_gate = QFTFullGate(n, approximation_degree=self.qft_approx_degree(n)).inverse()
            self.compose(inv_qft_gate, y_bits, inplace=True)

    def c_add_quantum(
        self,
        control_reg: QuantumRegister | list[Qubit] | Qubit,
        x_reg: QuantumRegister | list[Qubit],
        y_reg: QuantumRegister | list[Qubit],
        A: int = 1,
        include_QFT: bool = True,
    ) -> None:
        """
        Controlled version of 'add_quantum'.
        """
        control_bits = self.get_qubits(control_reg)
        x_bits = x_reg[:]
        y_bits = y_reg[:]
        m = len(x_bits)
        n = len(y_bits)

        if include_QFT:
            # QFT
            qft_gate = QFTFullGate(n, approximation_degree=self.qft_approx_degree(n))
            self.compose(qft_gate, y_bits, inplace=True)
        for i in range(m):
            for j in range(n):
                if i + j < n:  # The phase rotation is trivial if i+j>=n.
                    theta = 2 * np.pi * A * (2 ** (i + j - n))
                    self.mcp(theta, control_bits + [x_bits[i]], y_bits[j])
        if include_QFT:
            # Inverse QFT
            inv_qft_gate = QFTFullGate(n, approximation_degree=self.qft_approx_degree(n)).inverse()
            self.compose(inv_qft_gate, y_bits, inplace=True)

    def c_add_quantum_optimized_depth(
        self,
        control_reg: QuantumRegister | list[Qubit] | Qubit,
        x_reg: QuantumRegister | list[Qubit],
        y_reg: QuantumRegister | list[Qubit],
        A: int = 1,
        include_QFT: bool = True,
    ) -> None:
        """
        Controlled version of 'add_quantum', with modified circuit for depth
        optimization, following https://arxiv.org/pdf/1207.0511.
        """
        control_bits = self.get_qubits(control_reg)
        x_bits = x_reg[:]
        y_bits = y_reg[:]
        m = len(x_bits)
        n = len(y_bits)

        if include_QFT:
            # QFT
            qft_gate = QFTFullGate(n, approximation_degree=self.qft_approx_degree(n))
            self.compose(qft_gate, y_bits, inplace=True)
        for i in range(m):
            for j in range(n):
                if i + j <= n:  # The phase rotation is trivial if i+j>n.
                    self.cp(np.pi * A * (2 ** (i + j - n)), x_bits[i], y_bits[j])
        for i in range(m):
            self.mcx(control_bits, x_bits[i])
        for i in range(m):
            for j in range(n):
                if i + j <= n:  # The phase rotation is trivial if i+j>n.
                    self.cp(-np.pi * A * (2 ** (i + j - n)), x_bits[i], y_bits[j])
        for i in range(m):
            self.mcx(control_bits, x_bits[i])
        for j in range(n):
            phase = sum([np.pi * A * (2 ** (i + j - n)) for i in range(m)])
            self.mcp(phase, control_bits, y_bits[j])
        if include_QFT:
            # Inverse QFT
            inv_qft_gate = QFTFullGate(n, approximation_degree=self.qft_approx_degree(n)).inverse()
            self.compose(inv_qft_gate, y_bits, inplace=True)
