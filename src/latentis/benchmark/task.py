from abc import abstractmethod
from typing import Any

from latentis.serialize.io_utils import SerializableMixin


class Task(SerializableMixin):
    def __init__(self) -> None:
        self._result = None

    @abstractmethod
    def _run(self) -> Any:
        raise NotImplementedError

    def run(self, force: bool = False) -> Any:
        if self._result is None or force:
            self._result = self._run()

        return self._result
