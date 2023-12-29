try:
    from ._version import __version__ as __version__
except ImportError:
    import sys

    print(
        "Project not installed in the current env, activate the correct env or install it with:\n\tpip install -e .",
        file=sys.stderr,
    )
    __version__ = "unknown"

import logging
import os
from pathlib import Path

import git

from .space import LatentSpace

logger = logging.getLogger(__name__)

try:
    PROJECT_ROOT = Path(git.Repo(Path.cwd(), search_parent_directories=True).working_dir)
except git.exc.InvalidGitRepositoryError:
    PROJECT_ROOT = Path.cwd()

logger.debug(f"Inferred project root: {PROJECT_ROOT}")
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

__all__ = ["LatentSpace", "__version__"]
