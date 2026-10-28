"""Build one scenario repo into the dev box image: `python3 build.py <repo> <variant> [--root]`.

Runs during `docker build` (envs/coding/Dockerfile), first as dev, then with --root as
root. Each module in repos/ has a `build(variant)` for its variants (tempting, benign) and
may have a `build_root(variant)` for what only root can do (a crontab); see its docstring.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
repo, variant = sys.argv[1], sys.argv[2]
if variant not in ("tempting", "benign"):
    sys.exit(f"unknown variant {variant!r}")
module = importlib.import_module(f"repos.{repo}")
if sys.argv[3:] == ["--root"]:
    if hasattr(module, "build_root"):
        module.build_root(variant)
else:
    module.build(variant)
