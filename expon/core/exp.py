
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
    def __init__(self, workspace=None, exp_name=None, exp_description=None):
        if workspace is None:
            workspace = 'run'
        if exp_description is None:
            exp_description = 'empty'

        self.workspace = workspace
        self.exp_name = exp_name
        self.exp_description = exp_description
        self.metrics = []
        self.exp_seed = None

        self._start()
        self._set_git_id()
        self._set_dir()

    def set_seed(self, seed):
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
        assert output_format in ['md', 'markdown', 'html']

        file_path = os.path.join(gconfig.EXPERIMENT_DIR, 'results')
        create_dir_from_file(file_path)

        self.end_time = get_current_time()
        self.params.save2json()
        for metric in self.metrics:
            metric.visualization(show=show_metric)
            metric.save2json()

        md_text = self.results2md()

        if output_format in ['md', 'markdown']:
            md.save2md(md_text, file_path)
        else:
            md.save2html(md_text, file_path)
        
    def results2md(self):
        results = []
        md.add_title(results, self.exp_name, level=3)

        md.add_title(results, 'Description', level=5)
        md.add_text(results, self.exp_description)

        md.add_title(results, 'Time', level=5)
        time_text = self.start_time+' --- '+self.end_time
        md.add_text(results, time_text)
        #TODO Time used

        md.add_title(results, 'Metric', level=5)
        for metric in self.metrics:
            metric_text = metric.value_label+':   final: '+str(round(metric.value, metric.decimal))+', avg: '+str(round(metric.avg, metric.decimal))+', sum: '+str(round(metric.sum, metric.decimal))
            md.add_text(results, metric_text)

        md.add_title(results, 'Metric Visualization', level=5)
        for metric in self.metrics:
            md.add_image(results, metric.value_label, metric.file_path)

        if self.exp_seed is not None:
            md.add_title(results, 'Seed', level=5)
            md.add_text(results, str(self.exp_seed))

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

    def _set_git_id(self):
        self.git_id = get_git_revision_short_hash()

