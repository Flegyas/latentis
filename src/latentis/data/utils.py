from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
from transformers import BatchEncoding

from latentis.modules import LatentisModule
from latentis.serialize.io_utils import SerializableMixin


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

    @property
    def x(self):
        return self._x2y.keys()

    @property
    def y(self):
        return self._y2x.keys()

    def __repr__(self) -> str:
        return repr(self._x2y)


def default_collate(
    samples: Sequence,
    feature: str,
    model: LatentisModule,
    id_column: str = None,
) -> BatchEncoding:
    from latentis.data.processor import _ID_COLUMN

    id_column = id_column or _ID_COLUMN
    batch = model.pre_encode(samples, feature=feature)
    batch[id_column] = [sample[id_column] for sample in samples]

    return batch
