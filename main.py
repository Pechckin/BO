from rl.train.runner import Runner
import random 
import numpy as np
import torch
import warnings

CONFIG_PATH = "rl/config"


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    Runner(CONFIG_PATH).run()
