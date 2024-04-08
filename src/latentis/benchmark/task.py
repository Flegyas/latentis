from typing import Any


class Task:
    def __init__(self) -> None:
        self._result = None

    def _run(self) -> Any:
        pass

    def run(self, force: bool = False) -> Any:
        if self._result is None or force:
            self._result = self._run()

        return self._result
