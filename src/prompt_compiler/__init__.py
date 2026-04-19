"""Compatibility alias for the legacy `prompt_compiler` package name."""

import sys
from importlib import import_module

_pkg = import_module("prompt_transpiler")

globals().update(_pkg.__dict__)
__path__ = list(_pkg.__path__)

sys.modules[__name__] = _pkg
