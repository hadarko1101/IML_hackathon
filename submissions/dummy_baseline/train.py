#!/usr/bin/env python3

from pathlib import Path

import joblib

from model import ModelArchitecture


DATA_ROOT = Path("../../dataset")
OUTPUT = Path("weights.joblib")

def main() -> None:
    """
    Dummy training script.

    This does not train.
    It only creates the dummy CNN and saves its zero-initialized weights.
    The resulting model should always predict class 0.
    """

    model = ModelArchitecture(num_classes=20)

    state_dict = model.state_dict()
    joblib.dump(state_dict, OUTPUT)

    print(f"Saved dummy weights to {OUTPUT}")
    print("This dummy CNN always predicts class 0.")


if __name__ == "__main__":
    main()