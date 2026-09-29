"""Docker-only execution wrapper for Ko et al. (2019)."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Sequence

from src.ko_specificity.config import KoSpecificityConfig
from src.ko_specificity.io import (
    parse_predictions,
    read_input_csv,
    write_legacy_target_bundle,
    write_model_scores,
)

CommandRunner = Callable[..., subprocess.CompletedProcess]


def build_docker_command(
    config: KoSpecificityConfig, bundle_dir: Path, run_dir: Path
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{bundle_dir.resolve()}:/target:ro",
        "-v",
        f"{run_dir.resolve()}:/output",
        "-v",
        f"{config.glove_volume}:/artifacts:ro",
        config.image,
        "/opt/spectech/run_target.sh",
        "target",
    ]


def run_ko_specificity(
    config: KoSpecificityConfig,
    input_path: Path,
    output_path: Path,
    adaptation_context: str,
    *,
    command_runner: CommandRunner = subprocess.run,
    keep_run_dir: Path | None = None,
) -> Sequence[str]:
    """Run the diagnostic candidate only after an explicit gate authorization."""
    if not config.project_scoring_authorized:
        raise RuntimeError(
            "Ko project scoring is blocked: the official post-publication CPU path "
            "did not reproduce the released Twitter result, and no matching official "
            "paper-era CPU implementation or checkpoint is available"
        )
    rows = read_input_csv(input_path)
    if len(rows) < 50:
        raise ValueError(
            "The official train.py indexes target adaptation rows modulo 50; "
            "exact execution requires at least 50 target rows"
        )
    parent = keep_run_dir or output_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ko-target-", dir=parent) as temporary:
        run_root = Path(temporary)
        bundle_dir = run_root / "target"
        container_output = run_root / "output"
        bundle_dir.mkdir()
        container_output.mkdir()
        write_legacy_target_bundle(rows, bundle_dir)
        command = build_docker_command(config, bundle_dir, container_output)
        command_runner(command, check=True)
        predictions_path = container_output / "predictions.txt"
        scores = parse_predictions(predictions_path, len(rows))
        write_model_scores(output_path, rows, scores, config, adaptation_context)
        if keep_run_dir is not None:
            evidence = keep_run_dir / "evidence"
            evidence.mkdir(parents=True, exist_ok=True)
            for name in (
                "run_metadata.json",
                "model.sha256",
                "model.pickle",
                "predictions.txt",
                "train.log",
                "test.log",
            ):
                source = container_output / name
                if source.exists():
                    shutil.copy2(source, evidence / name)
    return command
