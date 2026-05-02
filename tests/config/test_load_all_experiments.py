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
from vrl.config.loader import (
    _REWARD_REQUIRED_KWARGS,
    build_algorithm_config,
    build_reward_config,
    load_config,
    optional_none,
    validate_reward_config,
    validate_training_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "configs" / "experiment"

_EXPECTED_ALGO_TYPE = {
    "grpo": GRPOConfig,
    "token_grpo": TokenGRPOConfig,
    "diffusion_dpo": DiffusionDPOConfig,
}

_EXPECTED_TRAIN_TARGET = {
    "cosmos_predict2_2b_grpo": "vrl.scripts.cosmos.train:train_cosmos_predict2_grpo",
    "janus_pro_1b_grpo": "vrl.scripts.janus_pro.train:train_janus_pro_grpo",
    "janus_pro_1b_ocr_grpo": "vrl.scripts.janus_pro.train:train_janus_pro_ocr_grpo",
    "nextstep_1_ocr_grpo": "vrl.scripts.nextstep_1.train:train_nextstep_1_ocr_grpo",
    "sd3_5_ocr_grpo": "vrl.scripts.sd3_5.train:train_sd3_5_grpo",
    "wan_2_1_14b_grpo": "vrl.scripts.wan_2_1.train:train_wan_2_1_grpo",
    "wan_2_1_1_3b_dpo": "vrl.scripts.wan_2_1.train_dpo:train_wan_2_1_dpo",
    "wan_2_1_1_3b_grpo": "vrl.scripts.wan_2_1.train:train_wan_2_1_grpo",
    "wan_2_1_1_3b_multi_reward_grpo": "vrl.scripts.wan_2_1.train:train_wan_2_1_grpo",
    "wan_2_1_1_3b_ocr_grpo": "vrl.scripts.wan_2_1.train:train_wan_2_1_grpo",
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


def test_no_experiment_named_training_wrappers() -> None:
    """Experiment names belong in YAML, not Python module names."""
    import re

    scripts_dir = REPO_ROOT / "vrl" / "scripts"
    experiment_names = set(_experiment_names())
    offenders: list[str] = []
    default_pat = re.compile(r'default_config\s*=\s*"experiment/([^"]+)"')
    for path in scripts_dir.rglob("*.py"):
        if path.stem in experiment_names:
            offenders.append(str(path.relative_to(REPO_ROOT)))
        if default_pat.search(path.read_text()):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, "remove experiment-specific script wrappers:\n" + "\n".join(offenders)


@pytest.mark.parametrize("name", _experiment_names())
def test_unified_train_entrypoint_dispatches_every_experiment(name: str) -> None:
    from vrl.scripts.train import resolve_train_target

    cfg = load_config(f"experiment/{name}")
    assert resolve_train_target(cfg).import_path == _EXPECTED_TRAIN_TARGET[name]


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
    with pytest.raises(ValueError, match=r"unknown algorithm\.kind"):
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


# ---------------------------------------------------------------------------
# SPRINT patch 3 Phase 7: validation gates and grep audits.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _experiment_names())
def test_validate_training_config_passes_for_all_active_experiments(
    name: str,
) -> None:
    """Every active experiment YAML must pass ``validate_training_config``.

    This is the top-level fail-fast contract: if any required path goes
    missing in the merged config, this gate flags the regression before
    runtime.
    """
    cfg = load_config(f"experiment/{name}")
    validate_training_config(cfg)


def test_aesthetic_kwargs_required() -> None:
    """Removing ``reward.kwargs.aesthetic.model_name`` from an aesthetic
    experiment must fail ``validate_reward_config`` with a clear message.
    """
    cfg = load_config("experiment/wan_2_1_1_3b_grpo")
    # Sanity: the aesthetic component is positive in this experiment.
    assert float(cfg.reward.components.get("aesthetic", 0.0)) > 0
    del cfg.reward.kwargs.aesthetic["model_name"]
    with pytest.raises(ValueError) as excinfo:
        validate_reward_config(cfg)
    msg = str(excinfo.value)
    assert "aesthetic" in msg and "model_name" in msg, msg


def test_pickscore_kwargs_required() -> None:
    """Removing ``reward.kwargs.pickscore.model_name`` from the Janus general
    experiment must fail ``validate_reward_config``.
    """
    cfg = load_config("experiment/janus_pro_1b_grpo")
    assert float(cfg.reward.components.get("pickscore", 0.0)) > 0
    del cfg.reward.kwargs.pickscore["model_name"]
    with pytest.raises(ValueError) as excinfo:
        validate_reward_config(cfg)
    msg = str(excinfo.value)
    assert "pickscore" in msg and "model_name" in msg, msg


@pytest.mark.parametrize("name", _experiment_names())
def test_active_experiments_have_no_legacy_ocr_debug_dir(name: str) -> None:
    """Wave 1 migrated all active experiments away from ``reward.ocr_debug_dir``
    onto ``reward.kwargs.ocr.debug_dir``. This test guards against regression.
    """
    cfg = load_config(f"experiment/{name}")
    if "reward" not in cfg:
        return
    assert "ocr_debug_dir" not in cfg.reward, (
        f"{name}: legacy reward.ocr_debug_dir reappeared — migrate to "
        "reward.kwargs.ocr.debug_dir"
    )


@pytest.mark.parametrize("name", _experiment_names())
def test_build_reward_config_returns_explicit_kwargs_for_each_positive_component(
    name: str,
) -> None:
    """For every model-backed reward with weight > 0, ``build_reward_config``
    must surface the required kwargs subkeys explicitly.
    """
    cfg = load_config(f"experiment/{name}")
    if "reward" not in cfg:
        return
    weights, kwargs = build_reward_config(cfg)
    for component, required_subkeys in _REWARD_REQUIRED_KWARGS.items():
        if component not in weights:
            continue
        component_kwargs = kwargs.get(component, {})
        assert isinstance(component_kwargs, dict), (
            f"{name}: kwargs[{component!r}] must be a mapping, got "
            f"{type(component_kwargs).__name__}"
        )
        for subkey in required_subkeys:
            assert subkey in component_kwargs, (
                f"{name}: missing reward.kwargs.{component}.{subkey}"
            )


def test_validate_training_config_fails_when_trainer_output_dir_missing() -> None:
    cfg = load_config("experiment/wan_2_1_1_3b_grpo")
    del cfg.trainer["output_dir"]
    with pytest.raises(ValueError, match=r"trainer\.output_dir"):
        validate_training_config(cfg)


def test_validate_training_config_fails_when_actor_optim_lr_missing() -> None:
    cfg = load_config("experiment/wan_2_1_1_3b_grpo")
    del cfg.actor.optim["lr"]
    with pytest.raises(ValueError, match=r"actor\.optim\.lr"):
        validate_training_config(cfg)


def test_validate_training_config_fails_when_nextstep_noise_level_missing() -> None:
    """NextStep-1's continuous-token AR pipeline requires ``rollout.noise_level``."""
    cfg = load_config("experiment/nextstep_1_ocr_grpo")
    del cfg.rollout["noise_level"]
    with pytest.raises(ValueError, match=r"rollout\.noise_level"):
        validate_training_config(cfg)


def test_dpo_allows_explicit_null_max_train_samples() -> None:
    """DPO's ``data.max_train_samples`` legitimately opts in to ``null``.

    ``optional_none`` returns ``None`` only when YAML explicitly sets the
    key to ``null``; ``validate_training_config`` then accepts it.
    """
    cfg = load_config("experiment/wan_2_1_1_3b_dpo")
    cfg.data.max_train_samples = None
    assert optional_none(cfg, "data.max_train_samples") is None
    validate_training_config(cfg)


def test_dpo_fails_when_max_train_steps_missing() -> None:
    cfg = load_config("experiment/wan_2_1_1_3b_dpo")
    del cfg.trainer["max_train_steps"]
    with pytest.raises(ValueError, match=r"trainer\.max_train_steps"):
        validate_training_config(cfg)


# Bonus tests: grep audits per patch §"Phase 7" lines 412 / 419 / 427.

# Keys whose `.get(<key>, <literal_default>)` form would silently re-introduce
# experiment-value Python fallbacks. Any such call is a SPRINT violation.
_FORBIDDEN_GET_KEYS: frozenset[str] = frozenset({
    "total_epochs",
    "output_dir",
    "noise_level",
    "sample_batch_size",
    "mixed_precision",
    "train_batch_size",
    "gradient_accumulation_steps",
    "scale_lr",
    "use_adafactor",
    "prediction_type",
    "max_train_steps",
    "checkpointing_steps",
    "log_interval",
    "dtype",
    "rank",
    "alpha",
    "dropout",
    "rescale_to_unit",
    "max_text_length",
    "same_latent",
    "sde",
    "eval_only",
    "prompts_file",
    "seeds",
    "reference_image",
})

# Heuristic: a `.get()` call whose receiver looks like a runtime metrics /
# observation dict is allowed (it's not config slicing).
_OBSERVATION_RECEIVERS: tuple[str, ...] = (
    "metrics",
    "last_components",
    "aux",
    "components",
    "last",
    "reward_kwargs",
    "reward_weights",
)


def test_no_silent_config_fallbacks_in_scripts() -> None:
    """Grep audit: scripts must not call ``X.get("<config_key>", <literal>)``
    for any key in :data:`_FORBIDDEN_GET_KEYS` — that re-introduces the
    experiment-default fallback the SPRINT eliminates.

    The check tolerates: (a) calls whose second argument is a non-literal
    (variable / expression with parentheses) and (b) calls on observation
    dicts (heuristic: receiver name appears in :data:`_OBSERVATION_RECEIVERS`).
    """
    import re

    scripts_dir = REPO_ROOT / "vrl" / "scripts"
    pat = re.compile(
        r"""(?P<recv>[A-Za-z_][\w\.]*)\.get\(\s*['"](?P<key>[A-Za-z_][\w]*)['"]\s*,\s*(?P<dflt>[^)]*)\)""",
    )
    bad: list[str] = []
    for path in scripts_dir.rglob("*.py"):
        text = path.read_text()
        for m in pat.finditer(text):
            key = m.group("key")
            if key not in _FORBIDDEN_GET_KEYS:
                continue
            recv = m.group("recv")
            recv_tail = recv.rsplit(".", 1)[-1]
            if recv_tail in _OBSERVATION_RECEIVERS:
                continue
            dflt = m.group("dflt").strip()
            # If the default is clearly a non-literal expression — e.g. a bare
            # identifier or attribute access — let it slide. Literals trigger.
            if not _looks_like_literal(dflt):
                continue
            bad.append(
                f"{path.relative_to(REPO_ROOT)}: "
                f"{recv}.get('{key}', {dflt})",
            )
    assert not bad, (
        "scripts still contain silent config fallbacks; convert to "
        "vrl.config.loader.require / optional_none:\n"
        + "\n".join(bad)
    )


def _looks_like_literal(expr: str) -> bool:
    """Return True iff the expression is one of: bool, None, number, string,
    list/tuple of literals. Parsed by :func:`ast.literal_eval`.
    """
    import ast

    expr = expr.strip()
    if not expr:
        return False
    try:
        ast.literal_eval(expr)
    except (ValueError, SyntaxError):
        return False
    return True


def test_no_legacy_ocr_debug_dir_in_active_yamls() -> None:
    """Raw-text grep audit: ``reward.ocr_debug_dir`` must not appear in any
    active experiment YAML. Wave 1 migrated all of them onto
    ``reward.kwargs.ocr.debug_dir``.
    """
    bad: list[str] = []
    for path in EXPERIMENT_DIR.glob("*.yaml"):
        text = path.read_text()
        if "ocr_debug_dir" in text:
            bad.append(str(path.relative_to(REPO_ROOT)))
    assert not bad, (
        "active experiment YAMLs still reference legacy reward.ocr_debug_dir:\n"
        + "\n".join(bad)
    )
