from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_EXPERIMENT_MODES = {"A", "B", "baseline"}


@dataclass(frozen=True)
class ExperimentModeConfig:
    mode: str
    use_glyph_prior: bool
    use_retrieval_prior: bool
    use_content_encoder: bool


def normalize_experiment_mode(mode: str | None) -> str:
    value = (mode or "baseline").strip()
    if value.lower() == "a":
        return "A"
    if value.lower() == "b":
        return "B"
    if value.lower() == "baseline":
        return "baseline"
    raise ValueError(f"Unsupported experiment mode: {mode}. Expected A, B, or baseline.")


def resolve_experiment_mode(mode: str | None) -> ExperimentModeConfig:
    """Resolve module switches for each experiment.

    A:
        text condition + style condition only. No glyph encoder, no retrieval,
        no content encoder.
    B:
        retrieval-rendered content prior encoded into glyph tokens.
    baseline:
        external content image encoded into glyph tokens.
    """

    normalized = normalize_experiment_mode(mode)
    if normalized == "A":
        return ExperimentModeConfig(
            mode="A",
            use_glyph_prior=False,
            use_retrieval_prior=False,
            use_content_encoder=False,
        )
    if normalized == "B":
        return ExperimentModeConfig(
            mode="B",
            use_glyph_prior=False,
            use_retrieval_prior=True,
            use_content_encoder=True,
        )
    return ExperimentModeConfig(
        mode="baseline",
        use_glyph_prior=False,
        use_retrieval_prior=False,
        use_content_encoder=True,
    )


def apply_experiment_mode_to_config(config: dict[str, Any], mode: str | None) -> dict[str, Any]:
    resolved = resolve_experiment_mode(mode)
    config.setdefault("experiment", {})["mode"] = resolved.mode
    model_config = config.setdefault("model", {})
    model_config["mode"] = resolved.mode
    model_config["use_glyph_prior"] = resolved.use_glyph_prior
    model_config["use_retrieval_prior"] = resolved.use_retrieval_prior
    model_config["use_content_encoder"] = resolved.use_content_encoder
    return config


def experiment_mode_log_lines(mode: str | None) -> list[str]:
    resolved = resolve_experiment_mode(mode)
    lines = ["[Experiment Mode]", f"mode={resolved.mode}"]
    if resolved.mode == "A":
        lines.extend(["text_condition=True", "style_condition=True", "glyph_condition=False"])
    elif resolved.mode == "B":
        lines.extend(["text_condition=True", "style_condition=True", "glyph_condition=True", "retrieval=True"])
    else:
        lines.extend(["content_condition=True", "text_condition=True", "style_condition=True", "glyph_condition=True"])
    return lines


def print_experiment_mode(mode: str | None) -> None:
    print("\n".join(experiment_mode_log_lines(mode)))


def add_experiment_mode_arg(parser):
    parser.add_argument("--mode", choices=sorted(VALID_EXPERIMENT_MODES), default="baseline")
    return parser
