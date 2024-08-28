import os.path

from . import arma, io, plot, model, lag_llama, chronos


def get_model_path(fname):
    module_path = os.path.dirname(__file__)
    project_path = os.path.abspath(os.path.join(module_path, ".."))
    return os.path.join(project_path, "models", fname)
