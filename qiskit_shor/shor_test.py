from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_shor.shor import find_factor, find_order


def test_find_order() -> None:
    # Seed the simulator to make tests deterministic
    pm = generate_preset_pass_manager(optimization_level=1, seed_transpiler=31)
    sampler = StatevectorSampler(seed=42)

    N = 15
    A = 2
    # 2^4 = 16 = 15*1 + 1
    want_order = 4
    got_order = find_order(A, N, sampler, pm, max_tries=1)

    assert got_order == want_order, f"Got {got_order}, want {want_order}"

    N = 15
    A = 7
    # 7^4 = 2401 = 15*160 + 1
    want_order = 4
    got_order = find_order(A, N, sampler, pm, max_tries=1)

    assert got_order == want_order, f"Got {got_order}, want {want_order}"


def test_find_factor() -> None:
    # Seed the simulator to make tests deterministic
    pm = generate_preset_pass_manager(optimization_level=1, seed_transpiler=31)
    sampler = StatevectorSampler(seed=42)

    N = 15
    want_factor = 3
    got_factor = find_factor(N, sampler, pm, num_tries=1, order_finding_max_tries=1, seed=13)

    assert got_factor == want_factor, f"Got {got_factor}, want {want_factor}"
