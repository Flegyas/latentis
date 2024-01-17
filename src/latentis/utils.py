import logging
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Sequence

import dotenv
import numpy as np
import pandas as pd
import torch

from latentis.types import SerializableMixin

pylogger = logging.getLogger(__name__)


def get_env(env_name: str, default: Optional[str] = None) -> str:
    """Safely read an environment variable.

    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable

    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            message = f"{env_name} not defined and no default value is present!"
            pylogger.error(message)
            raise KeyError(message)
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            message = f"{env_name} has yet to be configured and no default value is present!"
            pylogger.error(message)
            raise ValueError(message)
        return default

    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """Load all the environment variables defined in the `env_file`.

    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    if env_file is None:
        env_file = dotenv.find_dotenv(usecwd=True)
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


@contextmanager
def environ(**kwargs):
    """Temporarily set the process environment variables.

    https://stackoverflow.com/a/34333710

    >>> with environ(PLUGINS_DIR=u'test/plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True

    >>> "PLUGINS_DIR" in os.environ
    False

    :type kwargs: dict[str, unicode]
    :param kwargs: Environment variables to set
    """
    old_environ = dict(os.environ)
    os.environ.update(kwargs)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min

# https://github.com/Lightning-AI/lightning/blob/f6a36cf2204b8a6004b11cf0e21879872a63f414/src/lightning/fabric/utilities/seed.py#L19


def _select_seed_randomly(min_seed_value: int = min_seed_value, max_seed_value: int = max_seed_value) -> int:
    return random.randint(min_seed_value, max_seed_value)  # noqa: S3


def seed_everything(seed: Optional[int] = None) -> int:
    r"""Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random.

    In addition, sets the following environment variables:
    - ``PL_GLOBAL_SEED``: will be passed to spawned subprocesses (e.g. ddp_spawn backend).

    Args:
        seed: the integer value seed for global random state in Lightning.
            If ``None``, will read seed from ``PL_GLOBAL_SEED`` env variable
            or select it randomly.

    """
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
            pylogger.warn(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                pylogger.warn(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        pylogger.warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    pylogger.info(f"Seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


class BiMap(SerializableMixin):
    def __init__(self, x: Sequence[str], y: Sequence[int]):
        assert len(x) == len(y), "x and y must have the same length"
        self._x2y: Dict[str, int] = {k: v for k, v in zip(x, y)}
        self._y2x: Dict[int, str] = {v: k for k, v in self._x2y.items()}

    def contains_x(self, x: str) -> bool:
        return x in self._x2y

    def contains_y(self, y: int) -> bool:
        return y in self._y2x

    def get_x(self, y: int) -> str:
        return self._y2x[y]

    def get_y(self, x: str) -> int:
        return self._x2y[x]

    def add(self, x: str, y: int):
        assert x not in self._x2y, f"X `{x}` already exists"
        assert y not in self._y2x, f"Y `{y}` already exists"

        self._x2y[x] = y
        self._y2x[y] = x

    def add_all(self, x: Sequence[str], y: Sequence[int]):
        assert len(x) == len(y), "x and y must have the same length"
        for x_i, y_i in zip(x, y):
            self.add(x_i, y_i)

    def __len__(self) -> int:
        assert len(self._x2y) == len(self._y2x)
        return len(self._x2y)

    def save_to_disk(self, target_path: Path):
        df = pd.DataFrame({"x": list(self._x2y.keys()), "y": list(self._x2y.values())})
        df.to_csv(target_path, sep="\t", index=False)

    @classmethod
    def load_from_disk(cls, path: Path) -> "BiMap":
        mapping = pd.read_csv(path, sep="\t")
        return cls(x=mapping["x"].tolist(), y=mapping["y"].tolist())

    def __repr__(self) -> str:
        return repr(self._x2y)
