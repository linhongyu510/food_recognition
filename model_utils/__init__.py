"""兼容层：将 legacy.model_utils 重新导出。"""

from importlib import import_module


def _export(module):
    names = getattr(module, "__all__", None)
    if names is None:
        names = [name for name in dir(module) if not name.startswith("_")]
    for name in names:
        globals()[name] = getattr(module, name)
    return names


_data = import_module("legacy.model_utils.data")
_model = import_module("legacy.model_utils.model")
_train = import_module("legacy.model_utils.train")

__all__ = []
for _module in (_data, _model, _train):
    __all__.extend(_export(_module))
