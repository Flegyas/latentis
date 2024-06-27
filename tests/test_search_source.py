from pathlib import Path
from typing import Mapping

import pytest
import torch

from latentis.serialize.disk_index import DiskIndex
from latentis.space import Space
from latentis.utils import seed_everything


def test_index(tmp_path: Path):
    seed_everything(42)

    index = DiskIndex(tmp_path / "test_index", item_class=Space)
    n_items: int = 10

    random_properties = [{"p1": str(torch.randn(1, 1).item()), "p2": "value2"} for _ in range(n_items)]

    for x, properties in zip(range(n_items), random_properties):
        fake_item = Space(
            torch.randn(1, 1),
            metadata=properties,
        )

        index.add_item(item=fake_item)

    # get item by properties with multiple matches
    items = index.get_items(p2="value2")
    assert len(index) == n_items == len(items)

    # add item with same key
    new_item_key = index.add_item(item=Space(torch.randn(1, 2), metadata={"a": 1, "b": 2}))
    assert len(index) == n_items + 1

    with pytest.raises(FileExistsError):
        index.add_item(item=Space(torch.randn(1, 2), metadata=properties))

    index.remove_item(item_key=new_item_key)
    assert len(index) == n_items

    index.remove_item(**properties)
    new_item_key = index.add_item(item=Space(torch.randn(1, 2), metadata=properties))
    assert len(index) == n_items

    # get item by key
    item = index.get_item(new_item_key)
    assert isinstance(item, Mapping)
    assert len(index) == n_items

    index.remove_item(**index.load_item(new_item_key).metadata)
    assert len(index) == n_items - 1

    # get item by properties
    item = index.load_item(p1=random_properties[1]["p1"])
    assert isinstance(item, Space)

    # get item by properties with multiple matches
    with pytest.raises(ValueError):
        index.get_item(p2="value2")

    # get item by properties with no matches
    with pytest.raises(KeyError):
        assert index.get_item(p2="value3")

    # get items by empty properties matches all items
    assert len(index.get_items()) == len(index)

    # also the load
    assert len(index.load_items()) == len(index)

    index_restored = DiskIndex.load_from_disk(tmp_path / "test_index")
    assert len(index_restored) == len(index)

    # also removing them all D:
    index.remove_items()
    assert len(index) == 0
    assert len(list(tmp_path.iterdir())) == 1

    # clear is now a no-op
    index.clear()
    assert len(index) == 0
    assert len(list(tmp_path.iterdir())) == 1

    # but it works indeed
    index.add_item(item=Space(torch.randn(1, 2), metadata=properties))
    index.clear()
    assert len(index) == 0
    assert len(list(tmp_path.iterdir())) == 1
