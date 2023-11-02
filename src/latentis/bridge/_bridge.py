from typing import Any, Sequence

from latentis.brick import Brick


class BrickSequence(Brick):
    def __init__(self, bricks: Sequence[Brick]) -> None:
        self.bricks = bricks
        name = f'[{",".join(str(brick) for brick in  bricks)}]'
        super().__init__(name)

    def fit(self, x: Any, save: bool = True):
        assert save, "BrickSequence does not support stateless fitting"
        for brick in self.bricks:
            brick.fit(x)
            x = brick.forward(x)

    def forward(self, x: Any) -> Any:
        for brick in self.bricks:
            x = brick.forward(x)
        return x

    def reverse(self, x: Any) -> Any:
        for brick in reversed(self.bricks):
            x = brick.reverse(x)
        return x
