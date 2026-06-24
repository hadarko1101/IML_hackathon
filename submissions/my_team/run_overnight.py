import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


TEAM_DIR = Path(__file__).resolve().parent
LOG_DIR = TEAM_DIR / "overnight_logs"

RUNS = [
    {
        "name": "balanced_resnet",
        "epochs": 30,
        "batch_size": 32,
        "steps_per_epoch": 200,
    },
    {
        "name": "wide_resnet",
        "epochs": 30,
        "batch_size": 32,
        "steps_per_epoch": 200,
    },
    {
        "name": "deep_resnet",
        "epochs": 30,
        "batch_size": 32,
        "steps_per_epoch": 200,
    },
]


def build_command(run: dict) -> list[str]:
    model_name = run["name"]
    command = [
        sys.executable,
        str(TEAM_DIR / "train.py"),
        "--model",
        model_name,
        "--epochs",
        str(run["epochs"]),
        "--batch-size",
        str(run["batch_size"]),
        "--output",
        str(TEAM_DIR / f"weights_{model_name}.joblib"),
        "--metrics-output",
        str(LOG_DIR / f"metrics_{model_name}.csv"),
        "--num-workers",
        "0",
    ]

    if run["steps_per_epoch"] is not None:
        command.extend(["--steps-per-epoch", str(run["steps_per_epoch"])])

    return command


def run_one(run: dict) -> None:
    model_name = run["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{timestamp}_{model_name}.log"

    print(f"\n=== Starting {model_name} ===", flush=True)
    print(f"Log: {log_path}", flush=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            build_command(run),
            cwd=TEAM_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=env,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"{model_name} failed with exit code {return_code}")

    print(f"=== Finished {model_name} ===", flush=True)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for run in RUNS:
        run_one(run)

    print("\nAll overnight runs finished.")
    print(f"Weights and logs are in: {TEAM_DIR}")


if __name__ == "__main__":
    main()
