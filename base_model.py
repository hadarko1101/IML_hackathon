from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path

from labels import (
    HF_INDEX_TO_NAME,
    HF_INDEX_TO_IDX,
    TARGET_HF_INDICES,
)


class BaseModel(ABC):
    """
    Base class for all hackathon submissions.

    Competitors must:
      1. Subclass this class in their model.py
      2. Implement load() to restore weights from a file
      3. Implement predict() to return class predictions

    Submission format:
      team_name/
        model.py     <- contains a class named BaseModel that subclasses this
        weights.pt   <- saved with torch.save(model.state_dict(), 'weights.pt')
    """

    @abstractmethod
    def load(self, weights_path: str) -> None:
        """Load model weights from a .pt file."""
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a batch of inputs.

        Args:
            x: Float tensor of shape (batch_size, 3, 224, 224),
               normalized with ImageNet mean (0.485, 0.456, 0.406)
               and std (0.229, 0.224, 0.225).

        Returns:
            Long tensor of shape (batch_size,) with predicted class indices (0–19).
            These indices are defined by this hackathon (not by ImageNet).
            The mapping from index → class name is in labels.py and labels.json.
        """
        pass


class ImageNetSubset(Dataset):
    def __init__(self, root: Path, split: str, transform=None):
        self.transform = transform
        self.samples = []

        split_root = root / split

        if not split_root.exists():
            raise FileNotFoundError(f"Folder not found: {split_root}")

        for hf_idx in sorted(TARGET_HF_INDICES):
            class_name = HF_INDEX_TO_NAME[hf_idx]
            class_dir = split_root / class_name

            if not class_dir.exists():
                raise FileNotFoundError(f"Class folder not found: {class_dir}")

            local_idx = HF_INDEX_TO_IDX[hf_idx]

            image_paths = []
            image_paths.extend(class_dir.glob("*.jpg"))
            image_paths.extend(class_dir.glob("*.jpeg"))
            image_paths.extend(class_dir.glob("*.JPEG"))
            image_paths.extend(class_dir.glob("*.png"))

            for img_path in sorted(image_paths):
                self.samples.append((img_path, local_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {split_root}")

        print(f"Loaded {len(self.samples)} images from {split_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label