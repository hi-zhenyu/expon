
import json
import numpy as np

from  ..utils import vis
from ..utils import gconfig
from ..utils.file import *

class Metric():
    def __init__(self, value_label, step_label=None, decimal=6):
        if step_label is None:
            step_label = 'step'

        self.value= 0
        self.avg = 0
        self.sum = 0
        self.step = 0
        self.history = []

        self.value_label = value_label
        self.step_label = step_label
        self.decimal = decimal

    def update(self, value, n=1):
        self.value = value
        self.step += n
        self.sum += value * n
        self.avg = self.sum / self.step
        self.history.append([self.step, self.value])

    def visualization(self, show=True, filename=None, save_dir=None):
        if filename is None:
            filename = self.value_label + '-' + self.step_label

        if '.png' not in filename:
            filename += '.png'

        if save_dir is None:
            create_dir(gconfig.EXPERIMENT_DIR)
            file_path = os.path.join(gconfig.EXPERIMENT_DIR, filename)
        else:
            create_dir(save_dir)
            file_path = os.path.join(save_dir, filename)
        
        history = np.array(self.history)
        self.file_path = vis.draw_line(history[:, 0], 
                                        history[:, 1], 
                                        file_path, 
                                        self.step_label, self.value_label, show)

    def save2json(self, filename=None):
        if filename is None:
            filepath = os.path.join(gconfig.EXPERIMENT_DIR, self.value_label+'.json')
        else:
            if '.json' not in filename:
                filename = filename + '.json'
            filepath = os.path.join(gconfig.EXPERIMENT_DIR, filename)

        create_dir(gconfig.EXPERIMENT_DIR)
        with open(filepath, 'a+') as f:
            output = {self.value_label+'-'+self.step_label: self.history}
            f.write(dict2jsonstr(output))
