"""Command-line dispatcher for the paper's numerical studies (Sections 6.2 to 6.5; 6.5 has two)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STUDIES = (
    "controlled-projection-family",
    "same-sample-selection",
    "seventy-s-same-sample",
    "whitened-calibration",
    "colored-70s",
)


def _repo_root() -> Path:
    current = Path.cwd().resolve()
    if (current / "experiments").is_dir() and (current / "pyproject.toml").is_file():
        return current
    raise RuntimeError("run the experiment dispatcher from the repository root")


def command_for(
    study: str,
    mode: str,
    output_root: Path,
    protocol: Path | None,
) -> list[str]:
    root = _repo_root()
    experiments = root / "experiments"
    python = sys.executable
    output_root = output_root.resolve()
    if study == "controlled-projection-family":
        command = [
            python,
            str(experiments / "controlled_projection_family/run_experiment.py"),
            "--output-dir",
            str(output_root),
        ]
        if mode == "smoke":
            command.append("--smoke")
        return command
    if study == "same-sample-selection":
        script = experiments / "same_sample_selection/run_experiment.py"
        default_protocol = script.parent / "protocol.json"
        return [python, str(script), f"--{mode}", "--protocol", str(protocol or default_protocol), "--output-root", str(output_root)]
    if study == "seventy-s-same-sample":
        script = experiments / "seventy_s_same_sample/run_experiment.py"
        default_protocol = script.parent / "protocol.json"
        runner_mode = "quick" if mode == "smoke" else "formal"
        return [python, str(script), f"--{runner_mode}", "--protocol", str(protocol or default_protocol), "--output-root", str(output_root)]
    if study == "whitened-calibration":
        script = experiments / "whitened_calibration/run_experiment.py"
        command = [python, str(script), "--output-root", str(output_root)]
        if protocol is not None:
            command += ["--protocol", str(protocol)]
        if mode == "smoke":
            command.append("--smoke")
        return command
    script = experiments / "colored_70s/run_experiment.py"
    command = [python, str(script), "--output-root", str(output_root)]
    if protocol is not None:
        command += ["--protocol", str(protocol)]
    if mode == "smoke":
        command.append("--smoke")
    return command

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=STUDIES)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    command = command_for(
        args.study, args.mode, args.output_root, args.protocol
    )
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
