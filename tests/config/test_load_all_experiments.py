"""Per SPRINT_config_yaml_unification.md Phase 6 + the patch SPRINT Phase 5:
every experiment YAML must load via ``vrl.config.loader`` and expose the keys
downstream drivers expect; ``build_algorithm_config`` must dispatch correctly;
``algorithm.kind`` vs legacy ``adv_estimator`` conflicts must fail fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vrl.algorithms.dpo import DiffusionDPOConfig
from vrl.algorithms.grpo import GRPOConfig
from vrl.algorithms.grpo_token import TokenGRPOConfig
from vrl.config.loader import build_algorithm_config, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "configs" / "experiment"

_EXPECTED_ALGO_TYPE = {
    "grpo": GRPOConfig,
    "token_grpo": TokenGRPOConfig,
    "diffusion_dpo": DiffusionDPOConfig,
}


def _experiment_names() -> list[str]:
    return sorted(p.stem for p in EXPERIMENT_DIR.glob("*.yaml"))


@pytest.mark.parametrize("name", _experiment_names())
def test_experiment_yaml_loads(name: str) -> None:
    cfg = load_config(f"experiment/{name}")
    # Every experiment must compose at least these layers via `defaults:`.
    assert "model" in cfg, f"{name} missing model.*"
    assert "trainer" in cfg, f"{name} missing trainer.*"
    assert "algorithm" in cfg, f"{name} missing algorithm.*"
    assert "data" in cfg, f"{name} missing data.* source"

    # model must have a path (required by every family driver).
    assert "path" in cfg.model, f"{name} missing model.path"
    assert "output_dir" in cfg.trainer, f"{name} missing trainer.output_dir"
    # algorithm.kind is the canonical dispatch field; adv_estimator is legacy.
    assert "kind" in cfg.algorithm, f"{name} missing algorithm.kind"


@pytest.mark.parametrize("name", _experiment_names())
def test_build_algorithm_config_dispatch(name: str) -> None:
    cfg = load_config(f"experiment/{name}")
    algo_cfg = build_algorithm_config(cfg)
    expected = _EXPECTED_ALGO_TYPE[str(cfg.algorithm.kind)]
    # Token-GRPO must be the exact subclass, not the parent.
    if expected is GRPOConfig:
        assert type(algo_cfg) is GRPOConfig, (
            f"{name}: expected exact GRPOConfig, got {type(algo_cfg).__name__}"
        )
    else:
        assert isinstance(algo_cfg, expected), (
            f"{name}: expected {expected.__name__}, got {type(algo_cfg).__name__}"
        )


def test_no_experiment_orphans() -> None:
    """If a script declares a default_config, the file must exist."""
    import re

    scripts_dir = REPO_ROOT / "vrl" / "scripts"
    pat = re.compile(r'default_config\s*=\s*"experiment/([^"]+)"')
    referenced: set[str] = set()
    for path in scripts_dir.rglob("*.py"):
        for m in pat.finditer(path.read_text()):
            referenced.add(m.group(1))

    available = set(_experiment_names())
    missing = referenced - available
    assert not missing, f"scripts reference missing experiment YAMLs: {missing}"


def test_kind_vs_adv_estimator_conflict_fails_fast() -> None:
    """When kind and adv_estimator disagree, build_algorithm_config must raise."""
    cfg = OmegaConf.create({"algorithm": {"kind": "grpo", "adv_estimator": "dpo"}})
    with pytest.raises(ValueError, match="conflicts"):
        build_algorithm_config(cfg)


def test_kind_only_works_without_adv_estimator() -> None:
    cfg = OmegaConf.create({"algorithm": {"kind": "grpo"}})
    out = build_algorithm_config(cfg)
    assert type(out) is GRPOConfig


def test_legacy_adv_estimator_only_works() -> None:
    """Legacy YAML with only adv_estimator must still resolve to a typed config."""
    cfg = OmegaConf.create({"algorithm": {"adv_estimator": "token_grpo"}})
    out = build_algorithm_config(cfg)
    assert isinstance(out, TokenGRPOConfig)


def test_unknown_kind_fails_fast() -> None:
    cfg = OmegaConf.create({"algorithm": {"kind": "qpo"}})
    with pytest.raises(ValueError, match="unknown algorithm.kind"):
        build_algorithm_config(cfg)


def test_scripts_have_no_manual_algorithm_config_construction() -> None:
    """Scripts must consume ``built['algorithm']``; manual ``TokenGRPOConfig(...)``
    or ``DiffusionDPOConfig(...)`` re-introduces the experiment-defaults problem.
    """
    import re

    scripts_dir = REPO_ROOT / "vrl" / "scripts"
    bad: list[str] = []
    pat = re.compile(r"\b(TokenGRPOConfig|DiffusionDPOConfig)\(")
    for path in scripts_dir.rglob("*.py"):
        text = path.read_text()
        for m in pat.finditer(text):
            # Allow imports / type aliases — only construction `Foo(...)` is bad.
            line_start = text.rfind("\n", 0, m.start()) + 1
            line = text[line_start:text.find("\n", m.end())]
            if "import" in line or " as " in line:
                continue
            bad.append(f"{path.relative_to(REPO_ROOT)}:{line.strip()[:80]}")
    assert not bad, "scripts construct algorithm configs manually:\n" + "\n".join(bad)


def test_scripts_have_no_eps_clip_get_fallback() -> None:
    """Scripts must not silently fall back to literal experiment defaults via
    ``algo.get('eps_clip', 0.2)``.
    """
    import re

    scripts_dir = REPO_ROOT / "vrl" / "scripts"
    pat = re.compile(r"""\.get\(['"](eps_clip|init_kl_coef|adv_clip_max|kl_estimator|mask_key|beta|sft_weight)['"]""")
    bad: list[str] = []
    for path in scripts_dir.rglob("*.py"):
        for m in pat.finditer(path.read_text()):
            bad.append(f"{path.relative_to(REPO_ROOT)}: {m.group(0)}")
    assert not bad, "scripts fall back to algorithm defaults:\n" + "\n".join(bad)


def test_scripts_have_no_experiment_argparse_defaults() -> None:
    """Audit grep: training scripts must not carry experiment-value defaults
    in argparse. Only ``--config`` and OmegaConf dotlist overrides allowed.
    """
    import re

    scripts_dir = REPO_ROOT / "vrl" / "scripts"
    bad: list[str] = []
    # Match `add_argument(...)` calls. The only allowed flags are --config and
    # the positional `overrides`. Anything else with a `default=` is a SPRINT
    # violation.
    add_arg_pat = re.compile(r"add_argument\((.*?)\)", re.DOTALL)
    for path in scripts_dir.rglob("*.py"):
        text = path.read_text()
        for m in add_arg_pat.finditer(text):
            args = m.group(1)
            if "--config" in args or '"overrides"' in args or "'overrides'" in args:
                continue
            if "default=" in args:
                bad.append(f"{path.relative_to(REPO_ROOT)}: {args.strip()[:80]}")
    assert not bad, "scripts carry experiment defaults in argparse:\n" + "\n".join(bad)
