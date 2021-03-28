from rl.train.runner import Runner
import random 
import numpy as np
import torch
import warnings
import multiprocessing
import json
from multiprocessing import Process


CONFIG_PATH = "rl/config"


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def run(value):
    config['tay_coef'] = value
    config['type'] = f'BO_EI_DIM8_TRNUM10_RC1_SC{value}'
    print(f'{value} start!!!, device={config["device"]}')
    Runner(config).run()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # достаем конфиг по указанному пути
    with open(f'{CONFIG_PATH}.json') as json_file:
        config = json.load(json_file)
    
    procs = []
    params = [1, 2, 5, 10, 100] #sc
    #params = [0.1, 0.5, 1, 2, 4] #rc
    #params = [2, 4, 6, 8, 10] #trnum
    #params = [2, 4, 6, 8, 10] #dim
    for index, number in enumerate(params):
        proc = Process(target=run, args=(number,))
        procs.append(proc)
        proc.start()
        
    for proc in procs:
        proc.join()
        

    
    

