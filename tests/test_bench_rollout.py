import importlib.util
from pathlib import Path


def _load_benchmark_module():
    module_path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_rollout.py"
    spec = importlib.util.spec_from_file_location("bench_rollout_test_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_benchmark_resets_execution_stats_after_warmup():
    bench = _load_benchmark_module()

    result = bench.run_benchmark(
        device="cpu",
        num_steps=2,
        batch_size=2,
        execution_mode="chunked",
        hidden_dim=64,
        num_layers=2,
        num_tokens=16,
        latent_dim=6,
        action_dim=8,
        warmup_runs=1,
        benchmark_runs=1,
        gpu_sample_interval_s=0.01,
    )

    assert result["execution_stats"]["transition_entities"] == 4
    assert result["execution_stats"]["transition_chunks"] == 2
