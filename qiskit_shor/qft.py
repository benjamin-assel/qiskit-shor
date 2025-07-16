from qiskit.circuit.library import QFTGate
from qiskit.synthesis import synth_qft_full


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
        self.definition = synth_qft_full(
            num_qubits=self.num_qubits,
            do_swaps=self.do_swaps,
            approximation_degree=self.approximation_degree,
            insert_barriers=self.insert_barriers,
        )
