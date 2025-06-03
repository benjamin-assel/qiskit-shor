

# qiskit-shor
Full implementation of Shor factoring algorithm with Qiskit SDK.

Python version: 3.13.2  
Packages versions are specified in `requirements.txt`.

Installation:
- Open a terminal window 
- Clone the repository on your local machine
- Navigate to the cloned directory
- [Optional] Create and activate a virtual environment
- Run `pip install -r requirements.txt`


## Disclaimer
The work presented here is a complete implementation of Shor's algorithm, which, in theory, can run to factorize large integers and prove the quantum advantage of Shor's algorithm. This would be the case if the quantum hardware was available, namely if one had access to a quantum processor with enough qubits and low enough quantum noise. 
As of 2025, the IBM quantum platform provides access to quantum processors, which allows to test the present code on a real quantum computer. However, the number of qubits available is limited, with up to a few hundreds of qubits, and, more importantly, the degree of quantum noise is still too high for applications such as Shor's factorization, even for small integers.
Therefore, the code presented here is more of an illustration of how Shor's algorithm works and how to implement it in Qiskit. It can be tested on noise-free similumators and it could be used in the future to test Shor's algorithm, as the hardware improves.


## Overview of Shor's algorithm
Shor's algorithm is a quantum algorithm which allows to factorize a composite integer $N$, in polynomial 
time complexity in $n = \log N$, with high probability.
The algorithm steps are:

0. If $N$ is even, 2 is a factor, or, if $N$ is a power of a prime $N = p^k$, with k >1, 
then $p$ is a factor and we are done.
1. Choose a random integer $1 < A < N$. 
If $gcd(A, N) > 1$, then $gcd(A, N)$ is a non-trivial factor of $N$ (lucky case) and we are done. 
Otherwise $A$ and $N$ are coprime (typical case).
2. Find the order of A in $Z_N$, i.e. the smallest integer $1 < r < N$ such that $A^r = 1 \mod N$, using the
phase estimation quantum algorithm.
3. If $r$ is odd, go to step 1 and choose another $A$.
Otherwise, compute $gcd(A^{r/2} - 1, N)$. If it is larger than 1, then it is a non-trivial factor of $N$. 
If not, start again in step 1.

This algorithm finds a factor of $N$ in polynomial time, with high probability. 

The order finding step 3 is a quantum search algorithm. It is probabilistic in nature, so one needs to repeat it until it succeeds. Its success rate depends on the 'precision' required, which gets reflected into the size of the quantum circuit.

Introductions to Shor's algorithm are easy to find on the web (e.g. on Wikipedia). 

Original paper:
Shor, P. W. (1999). *Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer*. SIAM review, 41(2), 303-332. https://arxiv.org/abs/quant-ph/9508027

## Qubit conventions
The implementation uses Qiskit conventions:
* Classical integers are represented by capital letters, e.g. $X$, $Y$, $A$.
* Quantum integers are represented by small letters, e.g. $x$, $y$.
A quantum integer $x$ corresponds to a quantum state $|x\rangle := |x_0\rangle |x_1\rangle ... |x_m\rangle$, 
with $x_k \in \{0, 1\}$, where $x = \sum_{k=0}^{m-1} x_k 2^k$. In binary string notation, it is the integer $x_m ... x_1 x_0$, so $x_0$ is the least significant bit (LSB), and it is ordered in the qubit register as [qubit_0, qubit_1, ..., qubit_m], where qubit_k is in the 
state $|x_k\rangle$. Often the quantum register in the state representing $x$ is called "x_reg".

## Modular arithmetics
The modular arithmetic circuit operations are implemented in `adder.py`, via the
`AdderCircuit` class methods.
These operations are implemented in Fourier space, leveraging the rotation gates, 
which are part of the basic set of gates in Qiskit, and the QFT gate to convert between
computational and Fourier spaces.
The implementation is based on the work of Beauregard: *Circuit for Shor's algorithm using 2n+ 3 qubits*. https://arxiv.org/abs/quant-ph/0205095. 

To create a circuit, use the `AdderCircuit` class, which is a subclass of `QuantumCircuit`.

```python
from qiskit_shor.adder import AdderCircuit

# Create a circuit with 5 qubits
qc1 = AdderCircuit(5)

# Create a circuit with 3 qubits and 3 classical bits 
q_reg = QuantumRegister(5)
c_reg = ClassicalRegister(2)
qc2 = AdderCircuit(q_reg, c_reg)
```
To perform an operation, use the methods of the `AdderCircuit` class.
Since we are dealing with finite size quantum registers, all operations, even the non-modular ones, 
are enforced modulo $2^k$, where $k$ is the size of the target quantum register, conventionally 
named $y$ in this package. One typically ensures that the size of the target register is large enough 
to contain the result of the operation.

```python
x_reg = QuantumRegister(3)
y_reg = QuantumRegister(4)
ancilla_reg = QuantumRegister(1)

qc = AdderCircuit(x_reg, y_reg, ancilla_reg)
# x -> x + 3
qc.add_classical(3, x_reg)
# y -> y + 6
qc.add_classical(6, y_reg)
# y -> y + x
qc.add_quantum(x_reg, y_reg)
# y -> y + 10*x
qc.add_quantum(x_reg, y_reg, A=10)
# x -> (x + 7) mod 9  
qc.add_classical_modulo(7, x_reg, ancilla_reg[0], N=9)
# y -> (y + 4*x) mod 9
qc.add_quantum_modulo(x_reg, y_reg, ancilla_reg[0], N=9, A=4)

z_reg = QuantumRegister(6)
# x -> 4*x mod 9
qc.multiply_modulo(
    A=4, x_reg=x_reg, y_reg=z_reg[:4], overflow_bit=z_reg[4], ancilla_bit=z_reg[5], N=9,
)
# (x, y) -> (x, y*(4^x) mod 9)
qc.exponentiate_modulo(A=4, x_reg=x_reg, y_reg=y_reg, ancilla_reg=z_reg, N=9)
```

Controlled operations are also supported.
```python
control_reg = QuantumRegister(1)
x_reg = QuantumRegister(3)
y_reg = QuantumRegister(4)
ancilla_reg = QuantumRegister(1)

qc = AdderCircuit(control_reg, x_reg, y_reg, ancilla_reg)
# Flip control bit
qc.x(control_reg[0])
# Controlled operation x -> x + 3
qc.c_add_classical(control_reg, 3, x_reg)
# Controlled operation x -> (x + 6) mod 9
qc.c_add_classical_modulo(control_reg, 6, x_reg, ancilla_reg[0],N=9)
# Controlled operation y -> y + x
qc.c_add_quantum(control_reg, x_reg, y_reg, ancilla_reg[0])
# Controlled operation y -> (y + 10*x) mod 9
qc.c_add_quantum_modulo(control_reg, x_reg, y_reg, ancilla_reg[0], N=9, A=10)

z_reg = QuantumRegister(6)
# Controlled operation x -> 4*x mod 9
qc.c_multiply_modulo(control_reg, 4, x_reg, z_reg[:4], z_reg[4], z_reg[5], N=9)
```
The control register input can be a register of several qubits, to implement a 
multi-qubit controlled operation. It can also be a single Qubit.

The available operations, their input qubits requirements and input state assumptions are described in `adder.py` 
(see method desxriptions).

## Shor factorization
The order finding circuit and Shor factorization algorithm are implemented in `shor.py`.
The main API functions are `find_order` and `find_factor`, which build the order 
finding circuit and run it on the provided quantum backend or simulator.
```python
from qiskit_shor.shor import find_order, find_factor

# Define your sampler and pass_manager
sampler = ...
pass_manager = ...

N = 15
A = 7
# Compute the order of A in Z_N, running the circuit 100 times. Return the order and the distribution of measurement outcomes.
order, distribution = find_order(
    A, N, sampler, pass_manager, num_shots=100, one_control_circuit=True,
)
# Compute a factor of N using Shor algortihm, trying 3 random values for A and running the circuit 100 times for each try.
factor = find_factor(
    N, sampler, pass_manager, num_tries=3, num_shots_per_trial=100, one_control_circuit=True,
)
```
The order finding circuit is implemented in two variants: the basic circuit using $4n+2$ qubits with measurements 
at the end of the circuit, and the "one-control" circuit using $2n+3$ qubits and control flow operations on one qubit.
These two variants are described in Beauregard's paper. They are toggled using the argument `one_control_circuit`.

Some examples of the code usage on simulators and real devices can be found in the `example.ipynb` notebook.

## Algorithm complexity

This implementation is not optimal in terms of number of qubits required, nor in terms of number of elementary gates (single qubit or two-qubit gates). It is arguably the simplest implementation of the modular operations needed to create the order-finding quantum circuit, in the Fourier Transform paradigm.

With $n := \lceil \log_2 N \rceil$, the basic order finding circuit requires $4n+2$ qubits, while the circuit using a single control qubit requires $2n+3$ qubits in total. The number of gates is $O(n^4)$ and the depth is $O(n^3)$.

## Testing
Unit tests can be run with `pytest`.
```
python -m pytest <TEST_FILE>.py
```

## Resources

IBM Quantum Learning: https://learning.quantum.ibm.com,  
Shor factorization course: https://learning.quantum.ibm.com/course/fundamentals-of-quantum-algorithms/phase-estimation-and-factoring  
IBM Quantum Platform: https://quantum.ibm.com