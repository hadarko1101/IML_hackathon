import joblib
import torch

from base_model import BaseModel
from model import ModelArchitecture


class Model(BaseModel):
    def __init__(self):
        self.net = ModelArchitecture(num_classes=20)

    def load(self, weights_path: str) -> None:
        state_dict = joblib.load(weights_path)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.net(x)
            preds = logits.argmax(dim=1)
        return preds