from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

from qiskit_shor.shor import find_factor, find_order


def test_find_order() -> None:
    aer_sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    aer_sampler = AerSampler(seed=5)  # Seed the sampler for deterministic test

    N = 15
    A = 2
    # 2^4 = 16 = 15*1 + 1
    want_order = 4
    # Case: Default circuit
    got_order, _ = find_order(A, N, aer_sampler, pm, num_shots=10)
    assert got_order == want_order, f"Got {got_order}, want {want_order}"
    # Case: One control circuit
    got_order, _ = find_order(A, N, aer_sampler, pm, num_shots=10, one_control_circuit=True)
    assert got_order == want_order, f"Got {got_order}, want {want_order}"

    N = 15
    A = 7
    # 7^4 = 2401 = 15*160 + 1
    want_order = 4
    # Case: Default circuit
    got_order, _ = find_order(A, N, aer_sampler, pm, num_shots=10)
    assert got_order == want_order, f"Got {got_order}, want {want_order}"
    # Case: One control circuit
    got_order, _ = find_order(A, N, aer_sampler, pm, num_shots=10, one_control_circuit=True)
    assert got_order == want_order, f"Got {got_order}, want {want_order}"


def test_find_factor() -> None:
    aer_sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    aer_sampler = AerSampler(seed=5)  # Seed the sampler for deterministic test

    N = 15
    want_factor = 3
    got_factor = find_factor(N, aer_sampler, pm, num_tries=1, num_shots_per_trial=1, seed=13)

    assert got_factor == want_factor, f"Got {got_factor}, want {want_factor}"
