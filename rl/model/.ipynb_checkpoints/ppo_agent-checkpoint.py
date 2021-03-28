from rl.model.model import PPOModel
import numpy as np

class PPOAgent:

    def __init__(self, model: PPOModel):
        self.model = model

    def action(self, state: np.ndarray) -> np.ndarray:
        return self.model.act(state)
