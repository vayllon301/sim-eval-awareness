"""The coding family: a coding agent fixes tickets in small repos on a real dev box, with
C0-C3 observers for the bash tool (PLAN.md, from N1). See docs/TESTBED.md, "Coding family".

Importing the package registers the `rawchat` model provider (coding/provider.py), so a
config may use `rawchat/<served-name>` models.
"""

from . import provider  # noqa: F401  (registers the rawchat model API)

__all__ = ["provider"]
