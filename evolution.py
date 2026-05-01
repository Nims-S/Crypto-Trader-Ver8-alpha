# legacy compatibility
import importlib as _il
_mod = _il.import_module('legacy.evolution')
for _k in dir(_mod):
    if not _k.startswith('_'):
        globals()[_k] = getattr(_mod, _k)
