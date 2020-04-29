

import pprint as pp
from typing import Any
from collections import OrderedDict

from ..utils.file import *
from ..utils import gconfig

class Params:
    def __init__(self):
        self._param_dict = OrderedDict()

    def load_json(self, filepath):
        if '.json' not in filepath:
            filepath += '.json'
        if not check_file(filepath):
            raise Exception(filepath + ' dose not exist!')
        with open(filepath, 'r') as f:
            file_dict = jsonstr2dict(f.read())
            for k, v in file_dict.items():
                if k in ['git_id', 'start_time']:
                    pass
                else:
                    setattr(self, k, v)

    def save2json(self, filename=None):
        if filename is None:
            filepath = os.path.join(gconfig.EXPERIMENT_DIR, 'params.json')
        else:
            if '.json' not in filename:
                filename = filename + '.json'
            filepath = os.path.join(gconfig.EXPERIMENT_DIR, filename)

        create_dir(gconfig.EXPERIMENT_DIR)
        with open(filepath, 'w') as f:
            f.write(dict2jsonstr(self._param_dict))


    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._param_dict[name] = value

    def __setitem__(self, key, value):
        key = str(key)
        self.__setattr__(key, value)

    def __getattr__(self, item):
        if item not in self._param_dict:
            raise AttributeError(item)
        return self._param_dict[item]

    def __getitem__(self, item):
        return self._param_dict[item]

    def __repr__(self):
        return "{}".format(self.__class__.__name__) + pp.pformat([(k, v) for k, v in self._param_dict.items()])