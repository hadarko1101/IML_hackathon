import torch
import torch.nn as nn


class ModelArchitecture(nn.Module):
    """
    Dummy CNN.

    This is a real torch CNN module, but it is intentionally useless:
    all weights are initialized to zero, so all logits are identical.
    argmax will always return class 0.
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8, num_classes),
        )

        self._initialize_dummy_weights()

    def _initialize_dummy_weights(self) -> None:
        """
        Force the network to predict class 0 for every image.
        """
        for param in self.parameters():
            nn.init.constant_(param, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits