from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit

from qiskit_shor.adder import AdderCircuit
from qiskit_shor.gate_utils import MAJGate, MCXMAJGate, MCXUMAGate, UMAGate


class RCAdderCircuit(AdderCircuit):
    """
    Quantum circuit implementing modular arithmetics used in the order-finding circuit of Shor's algorithm.
    Operations are constructed from basic additions using the Ripple-Carry algorithm from
    Cucarro et al https://arxiv.org/abs/quant-ph/0410184.
    This method requires ancilla qubits for additions. It uses only X, CX and CCX gates.
    """

    @staticmethod
    def get_qubits(reg: QuantumRegister | list[Qubit] | Qubit) -> list[Qubit]:
        """Returns the provided qubits input as a list of qubits."""
        if isinstance(reg, Qubit):
            return [reg]
        return reg[:]

    def bitwise_AND_classical(self, X: int, y_reg: QuantumRegister | list[Qubit]) -> None:
        """
        Performs a bitwise_AND operation with the classical positive integer X onto the quantum integer y.
        When y_reg is |0>_n, the operation encodes the classical integer X into the quantum register y_reg.
        Applied a second time with the same X, it restores y_reg to its original state (self-inverse operation).

        Operation: |y> -> |X & y >

        The operation is performed modulo 2^n, where n is the size of the y register.

        Arguments:
        - X: integer.
        - y_reg: quantum register or list of qubits.

        Gate count: O(n).
        """
        y_bits = y_reg[:]
        n = len(y_bits)
        assert X > 0, "X must be a positive integer."
        x_bits = [int(i) for i in bin(abs(X))[-1:1:-1]]  # least significant bit first.
        for i in range(min(n, len(x_bits))):
            if x_bits[i] == 1:
                self.x(y_bits[i])

    def add_quantum(
        self,
        x_reg: QuantumCircuit,
        y_reg: QuantumCircuit,
        ancilla_bit: Qubit,
        overflow_bit: Qubit | None = None,
    ) -> None:
        """
        Adds the quantum integer x to the quantum integer y in-place,
        using one (optional)overflow qubit and one ancilla qubit.

        Operation: |x>_n|y>_n|0>|0> -> |x>_n|y+x>_{n+1}|0>

        n is the size of the x and y register.
        If no overflow qubit it provided, the result is given modulo 2^n in the y register.

        Arguments:
        - x_reg: quantum integer (size n).
        - y_reg: quantum register (size n).
        - ancilla_bit: ancilla qubit.
        - [Optional] overflow_bit: overflow qubit.

        Assumptions:
        - The x and y register must have the same size.
        - The ancilla and overflow qubits are in the |0> state.

        Gate count: O(n).
        """
        x_bits = x_reg[:]
        y_bits = y_reg[:]
        n = len(y_bits)

        assert len(x_bits) == n, "The x and y register must have the same size."

        if n == 0:
            return

        maj_gate = MAJGate()
        uma_gate = UMAGate(RC_version=True)

        self.append(maj_gate, [ancilla_bit, y_reg[0], x_reg[0]])
        for i in range(n - 1):
            self.append(maj_gate, [x_reg[i], y_reg[i + 1], x_reg[i + 1]])
        if overflow_bit:
            self.cx(x_reg[n - 1], overflow_bit)
        for i in range(n - 1, 0, -1):
            self.append(uma_gate, [x_reg[i - 1], y_reg[i], x_reg[i]])
        self.append(uma_gate, [ancilla_bit, y_reg[0], x_reg[0]])

    def c_add_quantum(
        self,
        control_reg: QuantumRegister | list[Qubit] | Qubit,
        x_reg: QuantumRegister | list[Qubit],
        y_reg: QuantumRegister | list[Qubit],
        ancilla_bit: Qubit,
        overflow_bit: Qubit | None = None,
    ) -> None:
        """
        Controlled version of add_quantum_RC.
        """
        ctrl_bits = self.get_qubits(control_reg)
        x_bits = x_reg[:]
        y_bits = y_reg[:]
        n = len(x_bits)

        assert len(y_bits) == n, "The x and y register must have the same size."

        mcx_maj_gate = MCXMAJGate(num_ctrl_qubits=len(ctrl_bits))
        mcx_uma_gate = MCXUMAGate(num_ctrl_qubits=len(ctrl_bits))

        self.append(mcx_maj_gate, [*ctrl_bits, ancilla_bit, y_reg[0], x_reg[0]])
        for i in range(n - 1):
            self.append(mcx_maj_gate, [*ctrl_bits, x_reg[i], y_reg[i + 1], x_reg[i + 1]])
        if overflow_bit:
            self.mcx([*ctrl_bits, x_reg[n - 1]], overflow_bit)
        for i in range(n - 1, 0, -1):
            self.append(mcx_uma_gate, [*ctrl_bits, x_reg[i - 1], y_reg[i], x_reg[i]])
        self.append(mcx_uma_gate, [*ctrl_bits, ancilla_bit, y_reg[0], x_reg[0]])

    def add_classical(
        self,
        X: int,
        y_reg: QuantumRegister | list[Qubit],
        a_reg: QuantumRegister | list[Qubit],
    ) -> None:
        """
        Adds the classical integer X to the quantum integer y in-place, modulo 2^n.
        Requires n+1 ancilla qubits (n for encoding X).

        Operation: |y>_n|0>_{n+1} -> |y+X>_n|0>_{n+1}

        n is the size of the y register.

        Arguments:
        - X: classical integer.
        - y_reg: quantum register or list of qubits.
        - a_reg: ancilla quantum register or list of qubits.

        Assumptions:
        - The a register is in the |0>_{n+1} state.

        Gate count: O(n).
        """
        y_bits = y_reg[:]
        a_bits = a_reg[:]
        n = len(y_bits)
        assert len(a_bits) >= n + 1, f"The a register must have at least {n + 1} qubits."
        a_bit = a_bits[0]
        x_bits = a_bits[1 : n + 1]

        Xmod = X % (2**n)

        # Encode X into x_bits
        self.bitwise_AND_classical(Xmod, x_bits)
        self.add_quantum(x_bits, y_bits, a_bit)
        # Reset x_bits
        self.bitwise_AND_classical(Xmod, x_bits)

    def c_add_classical(
        self,
        control_reg: QuantumRegister | list[Qubit] | Qubit,
        X: int,
        y_reg: QuantumRegister | list[Qubit],
        a_reg: QuantumRegister | list[Qubit],
    ) -> None:
        """
        Controlled version of add_classical_RC.
        """
        y_bits = y_reg[:]
        a_bits = a_reg[:]
        n = len(y_bits)
        assert len(a_bits) >= n + 1, f"The a register must have at least {n + 1} qubits."
        a_bit = a_bits[0]
        x_bits = a_bits[1 : n + 1]

        Xmod = X % (2**n)

        # Encode X into x_bits
        self.bitwise_AND_classical(Xmod, x_bits)
        self.c_add_quantum(control_reg, x_bits, y_bits, a_bit)
        # Reset x_bits
        self.bitwise_AND_classical(Xmod, x_bits)
