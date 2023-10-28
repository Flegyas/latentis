from pathlib import Path

import numpy as np
import pytest
import torch

from latentis import LatentSpace
from latentis.index import DataType, Index, Similarity
from latentis.utils import seed_everything


def _assert_index_eq(index1: Index, index2: Index):
    assert sorted(index1.ids) == sorted(index2.ids)
    assert np.allclose(a=index1.get_vectors(ids=sorted(index1.ids)), b=index2.get_vectors(ids=sorted(index1.ids)))
    assert np.allclose(a=index1.get_vector(id=0), b=index2.get_vector(id=0))


@pytest.mark.parametrize("num_vectors", [2, 1_000])
@pytest.mark.parametrize("num_dimensions", [10, 256])
@pytest.mark.parametrize("similarity_fn", sorted(Similarity))
def test_index(
    num_vectors: int,
    num_dimensions: int,
    similarity_fn: Similarity,
    tmp_path: Path,
):
    seed_everything(seed=0)

    space = LatentSpace(
        vectors=torch.randn(num_vectors, num_dimensions, dtype=torch.double),
        name="space1",
        features={
            "label": torch.rand(num_vectors) > 0.5,
        },
    )
    index = space.to_index(similarity_fn=similarity_fn)

    # Test properties
    assert sorted(index.ids) == sorted(range(num_vectors))
    assert index.max_elements is not None
    assert len(index) == num_vectors == index.num_elements
    assert index.num_dimensions == num_dimensions
    assert index.similarity_fn == similarity_fn
    assert index.serialization_data_type == DataType.FLOAT32
    assert str(index)

    for i in range(num_vectors):
        assert i in index

    # Test bytes serialization
    assert index.as_bytes()

    # Test query
    q = index.get_vector(0)
    neighbor_ids, distances = index.query(q, k=num_vectors)
    result = index.get_vector(neighbor_ids[0])
    assert np.allclose(a=q, b=result, atol=1e-5, rtol=1e-5)
    assert len(distances) == len(neighbor_ids) == num_vectors

    # Test query with torch tensor
    q = torch.as_tensor(index.get_vector(0))
    neighbor_ids, distances = index.query(q, k=num_vectors)
    result = index.get_vector(neighbor_ids[0])
    assert np.allclose(a=q, b=result, atol=1e-5, rtol=1e-5)

    # Test consistency with manuallly created index
    new_index: Index = Index.create(
        similarity_fn=similarity_fn,
        num_dimensions=num_dimensions,
    )
    for x in space.vectors:
        new_index.add_item(vector=x)
    _assert_index_eq(index1=index, index2=new_index)

    new_index: Index = Index.create(
        similarity_fn=similarity_fn,
        num_dimensions=num_dimensions,
    )
    new_index.add_items(vectors=space.vectors)
    _assert_index_eq(index1=index, index2=new_index)

    # Test distance function
    assert index.get_distance(a=space.vectors[0], b=space.vectors[1]) == new_index.get_distance(
        a=space.vectors[0], b=space.vectors[1]
    )

    # Test serialization
    tmp_index = tmp_path / "index.bin"
    index.save(filename=tmp_index)
    index_loaded: Index = Index.load(filename=tmp_index)
    _assert_index_eq(index1=index, index2=index_loaded)

    # Test efficient deletion
    index_loaded.mark_deleted(id=0)
    with pytest.raises(expected_exception=RuntimeError):
        index_loaded.mark_deleted(id=0)
    with pytest.raises(expected_exception=RuntimeError):
        index_loaded.get_vector(id=0)
    index_loaded.unmark_deleted(id=0)
    a = index_loaded.get_vector(id=0)
    assert a is not None and len(a) == num_dimensions and np.allclose(a=a, b=index.get_vector(id=0))

    # Test resize
    index.max_elements
    index.resize(new_size=3 * num_vectors)
    assert index.max_elements == 3 * num_vectors
