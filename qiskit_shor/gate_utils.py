from __future__ import annotations

from qiskit.circuit import Gate
from qiskit.circuit.library import QFTGate
from qiskit.circuit.singleton import SingletonGate


class QFTFullGate(QFTGate):
    """QFTGate supporting all the arguments of synth_qft_full."""

    # Whether to add swap gates in the end, to obtain the full QFT transformation.
    do_swaps: bool = True
    # The degree of approximation 0 <= d < n (0 for no approximation). Phase rotations with angles smaller than π/2^{n-d} will be dropped.
    # See https://arxiv.org/abs/quant-ph/9601018 and https://arxiv.org/abs/quant-ph/0403071.
    approximation_degree: int = 0
    # Whether to insert biarrier gates in the circuit for better visualization.
    insert_barriers: bool = False

    def __init__(
        self,
        num_qubits: int,
        do_swaps: bool = True,
        approximation_degree: int = 0,
        insert_barriers: bool = False,
    ):
        super().__init__(num_qubits=num_qubits)
        assert 0 <= approximation_degree < num_qubits, (
            f"The approximation degree d must satisfy 0 <= d < {num_qubits}, got d={approximation_degree}."
        )
        self.do_swaps = do_swaps
        self.approximation_degree = approximation_degree
        self.insert_barriers = insert_barriers

    def _define(self):
        """Provide a specific decomposition of the QFTGate into a quantum circuit."""
        from qiskit.synthesis.qft import synth_qft_full

        self.definition = synth_qft_full(
            num_qubits=self.num_qubits,
            do_swaps=self.do_swaps,
            approximation_degree=self.approximation_degree,
            insert_barriers=self.insert_barriers,
        )


class MAJGate(SingletonGate):
    """Inplace majority gate.
        |a,b,c> -> |a+c,b+c,maj(a,b,c)>
    where addition is modulo 2 and maj(a,b,c)=ab+bc+ac,
    i.e. maj(a,b,c)=1 if at least two inputs are 1, maj(a,b,c)=0 otherwise."""

    def __init__(self, label: str | None = None):
        """Create a MAJ gate."""
        super().__init__("maj", 3, [], label=label)

    def _define(self):
        """
        gate maj a,b,c
        {
          cx c, b;
          cx c, a;
          ccx a, b, c;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit, QuantumRegister
        from qiskit.circuit.library.standard_gates import CCXGate, CXGate

        q = QuantumRegister(3, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (CXGate(), [q[2], q[1]], []),
            (CXGate(), [q[2], q[0]], []),
            (CCXGate(), [q[0], q[1], q[2]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def approx_control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
    ):
        """Define a MAJ gate with the first CX controlled. This is used
        in Ripple-Carry controlled addition."""
        return MAJGate(num_ctrl_qubits=num_ctrl_qubits, label=label, _base_label=self.label)


class MCXMAJGate(Gate):
    """Define a MAJ gate with the first CX gate controlled (-> MCX gate). This is used
    for Ripple-Carry controlled addition."""

    def __init__(
        self,
        num_ctrl_qubits: int,
        label: str | None = None,
    ):
        """Create new MCX-MAJ gate."""
        self._num_ctrl_qubits = num_ctrl_qubits
        super().__init__("mcx_maj", num_ctrl_qubits + 3, [], label=label)

    def _define(self):
        """
        gate mcx_maj k,a,b,c
        {
          mcx [*k,c], b;
          cx c, a;
          ccx a, b, c;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit, QuantumRegister
        from qiskit.circuit.library.standard_gates import CCXGate, CXGate, MCXGate

        k = self._num_ctrl_qubits
        q = QuantumRegister(k + 3, "q")
        qc = QuantumCircuit(q, name=self.name)
        mcx_gate = MCXGate(num_ctrl_qubits=k + 1)
        rules = [
            (mcx_gate, [*q[:k], q[k + 2], q[k + 1]], []),
            (CXGate(), [q[k + 2], q[k]], []),
            (CCXGate(), [q[k], q[k + 1], q[k + 2]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc


class UMAGate(SingletonGate):
    """Inplace UnMajority and Add (UMA) gate.
    |a,b,c> -> |a+c+ab,a+b+c+ab,c+ab>
    where addition is modulo 2.
    Combined with the majority gate, the UMA gate can be used to compute the parity of a set of qubits.
    UMA.MAJ|a,b,c> = |a,a+b+c,c>.
    """

    def __init__(self, RC_version: bool = False, label: str | None = None):
        """Create a UMA gate."""
        super().__init__("uma", 3, [], label=label)
        # When true, the UMA gate is defined with 3-CNOTs instead of 2-CNOTs,
        # for depth-optimization in Ripple-Carry addition.
        self._RC_version = RC_version

    def _define(self):
        """
        gate uma a,b,c
        {
          ccx a, b, c;
          cx c, a;
          cx a, b;
        }
        or RC_version:
        gate uma a,b,c
        {
          x b;
          cx a, b;
          ccx a, b, c;
          x b;
          cx c, a;
          cx c, b;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit, QuantumRegister
        from qiskit.circuit.library.standard_gates import CCXGate, CXGate, XGate

        q = QuantumRegister(3, "q")
        qc = QuantumCircuit(q, name=self.name)
        if self._RC_version:
            rules = [
                (XGate(), [q[1]], []),
                (CXGate(), [q[0], q[1]], []),
                (CCXGate(), [q[0], q[1], q[2]], []),
                (XGate(), [q[1]], []),
                (CXGate(), [q[2], q[0]], []),
                (CXGate(), [q[2], q[1]], []),
            ]
        else:
            rules = [
                (CCXGate(), [q[0], q[1], q[2]], []),
                (CXGate(), [q[2], q[0]], []),
                (CXGate(), [q[0], q[1]], []),
            ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc


class MCXUMAGate(Gate):
    """Define an UMA gate with the last CX gate controlled (-> MCX gate). This is used
    for Ripple-Carry controlled addition."""

    def __init__(
        self,
        num_ctrl_qubits: int,
        label: str | None = None,
    ):
        """Create new MCX-MAJ gate."""
        self._num_ctrl_qubits = num_ctrl_qubits
        super().__init__("mcx_uma", num_ctrl_qubits + 3, [], label=label)

    def _define(self):
        """
        gate mcx_uma k,a,b,c
        {
          ccx a, b, c;
          cx c, a;
          mcx [*k,a], b;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit, QuantumRegister
        from qiskit.circuit.library.standard_gates import CCXGate, CXGate, MCXGate

        k = self._num_ctrl_qubits
        q = QuantumRegister(k + 3, "q")
        mcx_gate = MCXGate(num_ctrl_qubits=k + 1)
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (CCXGate(), [q[k], q[k + 1], q[k + 2]], []),
            (CXGate(), [q[k + 2], q[k]], []),
            (mcx_gate, [*q[: k + 1], q[k + 1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc
