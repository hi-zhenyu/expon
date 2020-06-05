
import random
from collections import OrderedDict

import numpy as np
import torch

from ..utils.file import *
from ..utils.time import get_current_time
from ..utils.git import *
from ..utils import gconfig
from ..utils import md

class EXP():
    def __init__(self, workspace=None, exp_name=None, exp_description=None, check_git=True):
        if check_git:
           check_working_tree()

        if workspace is None:
            workspace = 'run'
        if exp_description is None:
            exp_description = 'empty'

        self.workspace = workspace
        self.exp_name = exp_name
        self.exp_description = exp_description
        self.metrics = []
        self.info = {}
        self.exp_seed = None

        self._start()
        self._set_git_id()
        self._set_dir()

    def add_info(self, info_dict):
        self.info.update(info_dict)

    def get_info(self, key):
        return self.info[key]
    
    def set_seed(self, seed=None):
        '''
        Set experiment seed (random, numpy and torch). 
        If seed is None then randomly choose 0~1000 as seed.
        '''

        if seed is None:
            seed = random.randint(0,1000)
        else:
            assert type(seed) == int

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        self.exp_seed = seed

    def save(self, output_format='md', show_metric=False):
        '''
        Save the params, metrics.
        '''
        assert output_format in ['md', 'markdown', 'html']

        file_path = os.path.join(self.experiment_dir, 'results')
        create_dir_from_file(file_path)

        self.end_time = get_current_time()
        # save params
        self.params.save2json()

        # save metrics
        for metric in self.metrics:
            if metric.history:
                if metric.draw:
                    metric.visualization(show=show_metric)
                metric.save2json()

        md_text = self.results2md()

        if output_format in ['md', 'markdown']:
            md.save2md(md_text, file_path)
        else:
            md.save2html(md_text, file_path)
        
        print('exp infomation saved to', self.experiment_dir)

    def results2md(self):
        results = []
        md.add_title(results, self.exp_name, level=3)

        md.add_title(results, 'Description', level=5)
        md.add_text(results, self.exp_description)

        md.add_title(results, 'Time', level=5)
        time_text = self.start_time+' --- '+self.end_time
        md.add_text(results, time_text)
        #TODO Time used

        if self.exp_seed is not None:
            md.add_title(results, 'Seed', level=5)
            md.add_text(results, str(self.exp_seed))
        
        if self.metrics: 
            md.add_title(results, 'Metric', level=5)
            for metric in self.metrics:
                metric_text = metric.value_label+':   final: '+str(round(metric.value, metric.decimal))+', avg: '+str(round(metric.avg, metric.decimal))+', sum: '+str(round(metric.sum, metric.decimal))
                md.add_text(results, metric_text)

        for metric in self.metrics:
            if metric.draw:
                md.add_image(results, metric.value_label, metric.file_path)

        if self.info:
            md.add_title(results, 'Info', level=5)
            for (key, value) in self.info.items():
                md.add_title(results, key, level=6)
                md.add_text(results, str(value))

        md.add_title(results, 'Params', level=5)
        md.add_code(results, dict2jsonstr(self.params._param_dict))

        return results

    def set_params(self, params):
        self.params = params

    def add_metric(self, metric):
        self.metrics.append(metric)

    def _start(self):
        self.start_time = get_current_time()
        if self.exp_name is None:
            self.exp_name = self.start_time

    def _set_dir(self):
        if self.workspace is None:
            gconfig.WORKSPACE_DIR = os.path.join(gconfig.ROOT_DIR, 'run')
        else:
            gconfig.WORKSPACE_DIR = os.path.join(gconfig.ROOT_DIR, self.workspace)

        if self.exp_name is None:
            gconfig.EXPERIMENT_DIR = os.path.join(gconfig.WORKSPACE_DIR, self.start_time)
        else:
            gconfig.EXPERIMENT_DIR = os.path.join(gconfig.WORKSPACE_DIR, self.exp_name)

        if check_dir(gconfig.EXPERIMENT_DIR):
            raise Exception(gconfig.EXPERIMENT_DIR + 'already exits!')

        self.work_dir = gconfig.WORKSPACE_DIR
        self.experiment_dir = gconfig.EXPERIMENT_DIR

    def _set_git_id(self):
        self.git_id = get_git_revision_short_hash()

