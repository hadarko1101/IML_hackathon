import argparse
import importlib.util
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

WEIGHTS_FILENAME = "weights.joblib"

REQUIRED_FILES = [
    "train.py",
    "model.py",
    "predict.py",
    WEIGHTS_FILENAME,
]


def fail(message: str):
    print(f"[FAIL] {message}")
    return False


def pass_check(message: str):
    print(f"[ OK ] {message}")
    return True


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_team_submission(team_dir: Path):
    print()
    print("=" * 70)
    print(f"Checking submission: {team_dir.name}")
    print("=" * 70)

    ok = True

    if not team_dir.exists():
        return fail(f"Submission folder does not exist: {team_dir}")

    if not team_dir.is_dir():
        return fail(f"Submission path is not a folder: {team_dir}")

    pass_check("Submission folder exists")

    for filename in REQUIRED_FILES:
        file_path = team_dir / filename
        if file_path.exists() and file_path.is_file():
            pass_check(f"Found {filename}")
        else:
            ok = fail(f"Missing required file: {filename}") and ok

    if not ok:
        return False

    model_path = team_dir / "model.py"
    predict_path = team_dir / "predict.py"
    weights_path = team_dir / WEIGHTS_FILENAME

    # Make project root importable for base_model.py / labels.py.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # Make this team folder importable, so predict.py can do:
    # from model import ModelArchitecture
    sys.path.insert(0, str(team_dir))

    # Prevent Python from accidentally reusing another team's model.py.
    sys.modules.pop("model", None)

    try:
        model_module = load_module(model_path, f"{team_dir.name}_model")
        pass_check("model.py imports successfully")

        if not hasattr(model_module, "ModelArchitecture"):
            return fail("model.py must define a class named ModelArchitecture")

        pass_check("Found ModelArchitecture class in model.py")

        predict_module = load_module(predict_path, f"{team_dir.name}_predict")
        pass_check("predict.py imports successfully")

        if not hasattr(predict_module, "Model"):
            return fail("predict.py must define a class named Model")

        pass_check("Found Model class in predict.py")

        model = predict_module.Model()
        pass_check("Model() can be constructed")

    except Exception as e:
        return fail(f"Could not load submission files: {e}")

    finally:
        sys.path.pop(0)
        sys.modules.pop("model", None)

    if not hasattr(model, "load") or not callable(model.load):
        return fail("Model must implement load(weights_path)")

    pass_check("Model has load(...) method")

    if not hasattr(model, "predict") or not callable(model.predict):
        return fail("Model must implement predict(x)")

    pass_check("Model has predict(...) method")

    try:
        model.load(str(weights_path))
        pass_check(f"{WEIGHTS_FILENAME} loads successfully")
    except Exception as e:
        return fail(
            f"Could not load {WEIGHTS_FILENAME}. "
            "Most likely, the weights file does not match the architecture in model.py.\n"
            f"Error: {e}"
        )

    try:
        x = torch.randn(4, 3, 224, 224)
        preds = model.predict(x)
        pass_check("predict(x) runs successfully")
    except Exception as e:
        return fail(f"predict(x) failed: {e}")

    if not isinstance(preds, torch.Tensor):
        return fail("predict(x) must return a torch.Tensor")

    pass_check("predict(x) returns a torch.Tensor")

    if preds.shape != torch.Size([4]):
        return fail(
            f"predict(x) must return shape [batch_size]. "
            f"Expected [4], got {list(preds.shape)}"
        )

    pass_check("Prediction tensor has correct shape [batch_size]")

    if not torch.is_floating_point(preds):
        pass_check("Predictions are integer-like")
    else:
        return fail("Predictions must be class indices, not probabilities/logits")

    if preds.min().item() < 0 or preds.max().item() > 19:
        return fail(
            "Predicted labels must be between 0 and 19. "
            f"Got min={preds.min().item()}, max={preds.max().item()}"
        )

    pass_check("Predicted labels are in range 0..19")

    print()
    print(f"[SUCCESS] {team_dir.name} passed all format checks.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "team_name",
        nargs="?",
        help="Name of the team folder inside submissions/. If omitted, checks all teams.",
    )
    args = parser.parse_args()

    if not SUBMISSIONS_DIR.exists():
        print(f"[FAIL] Could not find submissions folder: {SUBMISSIONS_DIR}")
        sys.exit(1)

    if args.team_name:
        team_dirs = [SUBMISSIONS_DIR / args.team_name]
    else:
        team_dirs = sorted(
            d for d in SUBMISSIONS_DIR.iterdir()
            if d.is_dir()
        )

    if not team_dirs:
        print("[FAIL] No submission folders found.")
        sys.exit(1)

    all_ok = True

    for team_dir in team_dirs:
        team_ok = check_team_submission(team_dir)
        all_ok = all_ok and team_ok

    print()
    print("=" * 70)

    if all_ok:
        print("[SUCCESS] All checked submissions passed.")
        sys.exit(0)
    else:
        print("[FAIL] At least one submission has problems.")
        sys.exit(1)


if __name__ == "__main__":
    main()
