#!/usr/bin/env python3
"""Run Torch notebook smoke checks without writing executed outputs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "target" / "notebook-smoke"
OPTIONAL_ARTIFACT_MARKERS = (
    "datasets/mps",
    "mnist_0_mps.safetensors",
    "mnist_3_mps.safetensors",
    "mnist_experimental_mps.safetensors",
)


@dataclass
class NotebookResult:
    notebook: str
    status: str
    seconds: float
    message: str


def iter_torch_notebooks() -> list[Path]:
    requested = os.environ.get("TN_NOTEBOOKS")
    if requested:
        return [ROOT / item.strip() for item in requested.split(",") if item.strip()]

    notebooks = []
    for path in sorted(ROOT.glob("*.ipynb")):
        name = path.name.lower()
        if "mlx" in name:
            continue
        notebooks.append(path)
    return notebooks


def classify_error(error: BaseException) -> tuple[str, str]:
    message = str(error).replace("\n", "\\n")
    if any(marker in message for marker in OPTIONAL_ARTIFACT_MARKERS):
        return "skipped", message
    return "failed", message


def run_notebook(path: Path, *, device: str, timeout: int) -> NotebookResult:
    started = time.monotonic()
    env = os.environ.copy()
    env["TN_TORCH_DEVICE"] = device
    env["TN_NOTEBOOK_SMOKE"] = "1"

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    old_env = os.environ.copy()
    os.environ.update(env)
    try:
        client.execute()
    except CellExecutionError as error:
        status, message = classify_error(error)
    except Exception as error:
        status, message = classify_error(error)
    else:
        status, message = "passed", ""
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    return NotebookResult(
        notebook=path.name,
        status=status,
        seconds=round(time.monotonic() - started, 3),
        message=message,
    )


def write_reports(results: list[NotebookResult], *, device: str) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tsv_path = REPORT_DIR / f"notebooks_{device}_smoke.tsv"
    json_path = REPORT_DIR / f"notebooks_{device}_smoke.json"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("notebook\tstatus\tseconds\tmessage\n")
        for result in results:
            f.write(
                f"{result.notebook}\t{result.status}\t{result.seconds}\t{result.message}\n"
            )
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(result) for result in results], f, indent=2)
        f.write("\n")
    return tsv_path, json_path


def main() -> int:
    device = os.environ.get("TN_TORCH_DEVICE", "auto")
    timeout = int(os.environ.get("TN_NOTEBOOK_TIMEOUT", "600"))
    results = []
    for path in iter_torch_notebooks():
        result = run_notebook(path, device=device, timeout=timeout)
        results.append(result)
        print(f"{result.notebook}\t{result.status}\t{result.seconds}", flush=True)
    tsv_path, json_path = write_reports(results, device=device)

    print(f"TSV report: {tsv_path}")
    print(f"JSON report: {json_path}")

    return 1 if any(result.status == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
