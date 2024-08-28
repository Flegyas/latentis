from pathlib import Path
from typing import Type

from latentis import PROJECT_ROOT
from latentis.correspondence import Correspondence
from latentis.nn import LatentisModule
from latentis.serialize.disk_index import DiskIndex
from latentis.space import Space

_NEXUS_DIR: Path = PROJECT_ROOT / "nexus"
_NEXUS_DIR.mkdir(exist_ok=True)


def _init_index(path: Path, item_class: Type) -> None:
    try:
        index = DiskIndex.load_from_disk(path=path)
    except FileNotFoundError:
        index = DiskIndex(root_path=path, item_class=item_class)
        index.save_to_disk()

    return index


correspondences_index: DiskIndex = _init_index(
    path=_NEXUS_DIR / "correspondences", item_class=Correspondence
)
space_index: DiskIndex = _init_index(path=_NEXUS_DIR / "spaces", item_class=Space)
decoders_index = _init_index(path=_NEXUS_DIR / "decoders", item_class=LatentisModule)
