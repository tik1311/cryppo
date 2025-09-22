import torch
from torch import nn
from typing import Any, Dict

from src.models.base_model import BaseModel

class AlwaysLongModel(BaseModel):
    def __init__(self, input_size: int, config: Any):
        super().__init__(input_size, config)
        self.long_action = 1  # assumes LONG action id

    def _create_model(self) -> nn.Module:
        return nn.Identity()

    def train(self, train_loader, val_loader=None, epochs=None, log_dir=None) -> Dict[str, Any]:
        self.is_trained = True
        return {"history": []}

    def predict(self, inputs) -> int:
        return self.long_action

    def evaluate(self, data_loader, metrics_list=None) -> Dict[str, float]:
        return {"action_distribution": {str(self.long_action): 1.0}}

    def select_action(self, state, evaluation: bool = False) -> int:
        return self.long_action

    def store_experience(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        return None
