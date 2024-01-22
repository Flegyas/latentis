import uuid
from pathlib import Path

import pytest
import torch

from latentis.data.disk_index import DiskIndex
from latentis.space import LatentSpace
from latentis.utils import seed_everything


def test_index(tmp_path: Path):
    seed_everything(42)

    index = DiskIndex(tmp_path / "test_index", item_class=LatentSpace)
    n_items: int = 10

    random_ids = [uuid.uuid4().hex for _ in range(n_items)]
    random_properties = [{"p1": str(torch.randn(1, 1).item()), "p2": "value2"} for _ in range(n_items)]

    for x, x_id, properties in zip(range(n_items), random_ids, random_properties):
        fake_item = LatentSpace(
            torch.randn(1, 1),
            space_id=x_id,
        )

        index.add_item(
            item=fake_item,
            item_key=f"test_{x}",
            properties=properties,
        )

    # get item by properties with multiple matches
    items = index.get_items_by_properties(p2="value2")
    assert len(index) == n_items == len(items)

    # add item with same key
    fake_item = LatentSpace(
        torch.randn(1, 1),
        space_id=random_ids[0],
    )

    with pytest.raises(KeyError):
        index.add_item(
            item=fake_item,
            item_key="test_0",
        )

    index.add_item(
        item=fake_item,
    )
    assert len(index) == n_items + 1
    index.remove_item_by_key("test_0")
    assert len(index) == n_items
    index.add_item(
        item=fake_item,
        item_key="test_0",
    )

    # get item by key
    item = index.get_item_by_key("test_0")
    assert isinstance(item, LatentSpace)
    assert item.space_id == random_ids[0]

    # get item by properties
    item = index.get_item_by_properties(p1=random_properties[1]["p1"])
    assert isinstance(item, LatentSpace)

    # get item by properties with multiple matches
    with pytest.raises(KeyError):
        index.get_item_by_properties(p2="value2")

    # get item by properties with no matches
    with pytest.raises(KeyError):
        index.get_item_by_properties(p2="value3")

    for x in range(n_items):
        index.remove_item_by_key(f"test_{x}")
    index.clear()
    assert len(index) == 0
    assert len(list(tmp_path.iterdir())) == 1
