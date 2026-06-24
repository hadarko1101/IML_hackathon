import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.main(x) + self.shortcut(x))


class ResNetClassifier(nn.Module):
    """
    Configurable from-scratch ResNet-style CNN for the 20-class task.

    Required behavior:
        input:  torch.Tensor of shape [batch_size, 3, height, width]
        output: torch.Tensor of shape [batch_size, 20]
    """

    def __init__(
        self,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        blocks_per_stage: tuple[int, int, int, int] = (2, 2, 2, 2),
        dropout: float = 0.3,
        num_classes: int = 20,
    ):
        super().__init__()

        c1, c2, c3, c4 = channels

        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            self._make_stage(c1, c1, blocks_per_stage[0], stride=1),
            self._make_stage(c1, c2, blocks_per_stage[1], stride=2),
            self._make_stage(c2, c3, blocks_per_stage[2], stride=2),
            self._make_stage(c3, c4, blocks_per_stage[3], stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c4, num_classes),
        )

        self._initialize_weights()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        blocks = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(num_blocks - 1):
            blocks.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        logits = self.classifier(x)
        return logits


class BalancedResNet(ResNetClassifier):
    def __init__(self, num_classes: int = 20):
        super().__init__(
            channels=(32, 64, 128, 256),
            blocks_per_stage=(2, 2, 2, 2),
            dropout=0.3,
            num_classes=num_classes,
        )


class WideResNet(ResNetClassifier):
    def __init__(self, num_classes: int = 20):
        super().__init__(
            channels=(48, 96, 192, 384),
            blocks_per_stage=(2, 2, 2, 2),
            dropout=0.35,
            num_classes=num_classes,
        )


class DeepResNet(ResNetClassifier):
    def __init__(self, num_classes: int = 20):
        super().__init__(
            channels=(32, 64, 128, 256),
            blocks_per_stage=(3, 3, 3, 3),
            dropout=0.35,
            num_classes=num_classes,
        )


MODEL_REGISTRY = {
    "balanced_resnet": BalancedResNet,
    "wide_resnet": WideResNet,
    "deep_resnet": DeepResNet,
}

DEFAULT_MODEL_NAME = "balanced_resnet"


def build_model(model_name: str = DEFAULT_MODEL_NAME, num_classes: int = 20) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {model_name!r}. Available models: {available}")
    return MODEL_REGISTRY[model_name](num_classes=num_classes)


class ModelArchitecture(BalancedResNet):
    """
    Grader-facing default architecture.

    For the final submission, weights.joblib must come from this architecture.
    The overnight experiment script saves non-default variants separately.
    """

    pass
