import os

from .file import get_root_path

import pkg_resources

DEFAULT_CSS = pkg_resources.resource_filename('expon', 'asset/default.css')

GITHUB_CSS = pkg_resources.resource_filename('expon', 'asset/github.css')

CURRENT_DIR = get_root_path()

# default work directory ./EXP
ROOT_DIR = os.path.join(CURRENT_DIR, 'EXP')

# default work directory ./EXP/run
WORKSPACE_DIR = os.path.join(ROOT_DIR, 'run')

# experiment directory ./EXP/run/experiment
EXPERIMENT_DIR = None
