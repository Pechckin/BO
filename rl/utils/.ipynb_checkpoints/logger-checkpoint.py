from datetime import date
import torch
import numpy as np
from datetime import datetime

import json


class Logger(object):
    def __init__(self, save_dir):
        self.store = dict()
        self.save_dir = save_dir
        self.store['init_time'] = datetime.now().strftime("%Y%B%d")

    def log(self, stuff, type_):
        for name, value in stuff.items():
            if name not in self.store:
                self.store[name] = []
            self.store[name].append(value)
            

    def save(self):
        name_components = f'config_{self.store["init_time"]}_{self.store["type"]}_{self.store["env"]}'
        with open(f'{self.save_dir}/{name_components}.json', 'w') as outfile:
            json.dump(self.store, outfile, indent=4)
            
    def save_model(self, name, model):
        name_components = f'config_{self.store["init_time"]}_{self.store["type"]}_{self.store["env"]}'
        torch.save(model.state_dict(), f'{self.save_dir}/{name_components}_{name}')
