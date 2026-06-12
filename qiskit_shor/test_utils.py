from qiskit.primitives import StatevectorSampler
from qiskit.primitives.containers.sampler_pub_result import SamplerPubResult
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def run_simulation(qc, num_shots: int = 1) -> SamplerPubResult:
    pm = generate_preset_pass_manager(optimization_level=1)
    qc_isa = pm.run(qc)
    sampler = StatevectorSampler()
    result = sampler.run([qc_isa], shots=num_shots).result()[0]

    return result
