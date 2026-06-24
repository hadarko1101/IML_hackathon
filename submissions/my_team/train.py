from pathlib import Path

import joblib
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms

from base_model import ImageNetSubset
from model import ModelArchitecture


DATA_ROOT = Path("../../dataset")
OUTPUT = Path("weights.joblib")

IMAGE_SIZE = None
BATCH_SIZE = None



def main():
    """
    Full training pipeline.

    This script must create weights.joblib.
    """

    # TODO: load dataset (you might want to use ImageNetSubset)
    # TODO: create your model

    # TODO: save trained model weights to weights.joblib
    joblib.dump(model.state_dict(), "weights.joblib")
    print("Saved trained weights.joblib")

    raise NotImplementedError("TODO: implement main training pipeline")


if __name__ == "__main__":
    main()