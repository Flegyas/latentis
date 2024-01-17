from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from latentis.space import LatentSpace
from latentis.space.search import SearchIndex, SearchMetric
from latentis.utils import seed_everything


def _assert_index_eq(index1: SearchIndex, index2: SearchIndex):
    # assert sorted(index1.keys) == sorted(index2.keys)
    assert len(index1) == len(index2)
    assert np.allclose(
        a=index1.get_vectors(offsets=list(range(len(index1)))),
        b=index2.get_vectors(offsets=list(range(len(index1)))),
    )
    # assert index1.key2offset == index2.key2offset
    assert np.allclose(a=index1.get_vector(offset=0), b=index2.get_vector(offset=0))


@pytest.mark.parametrize("num_vectors", [2, 1_000])
@pytest.mark.parametrize("num_dimensions", [10, 256])
@pytest.mark.parametrize("metric_fn", sorted(SearchMetric, key=lambda x: x.value))
def test_index(
    num_vectors: int,
    num_dimensions: int,
    metric_fn: SearchMetric,
    tmp_path: Path,
):
    seed_everything(seed=0)

    space = LatentSpace(
        vector_source=torch.randn(num_vectors, num_dimensions, dtype=torch.double),
        name="space1",
    )
    index = space.to_index(metric_fn=metric_fn, keys=[str(i) for i in range(num_vectors)])

    # Test properties
    # assert sorted(index.ids) == sorted(range(num_vectors))
    # assert index.max_elements is not None
    assert len(index) == num_vectors == index.num_elements
    assert index.num_dimensions == num_dimensions
    assert index._metric_fn == metric_fn
    # assert index.storage_data_type == DataType.FLOAT32
    assert str(index)

    for i in range(num_vectors):
        assert str(i) in index

    # Test bytes serialization
    # assert index.as_bytes()

    # Test query
    q = index.get_vector(0)
    assert torch.equal(torch.as_tensor(q), index.get_vector(0, return_tensors=True))

    neighbor_ids, distances = index.search_knn(query_vectors=q, k=num_vectors)
    search_result = index.search_knn(query_offsets=[0], k=num_vectors)
    assert np.allclose(a=search_result.distances, b=distances)
    assert np.allclose(search_result.offsets, neighbor_ids)

    dict_result = search_result.asdict()
    assert isinstance(dict_result, dict)
    assert np.allclose(dict_result["offsets"], neighbor_ids)
    assert np.allclose(a=dict_result["distances"], b=distances)

    result = index.get_vector(int(neighbor_ids[0]))
    assert np.allclose(a=q, b=result, atol=1e-5, rtol=1e-5)
    assert len(distances) == len(neighbor_ids) == num_vectors

    # Test query with torch tensor
    q = torch.as_tensor(index.get_vector(0))
    q_tensor = index.get_vector(0, return_tensors=True)
    assert isinstance(q_tensor, torch.Tensor)
    assert torch.allclose(q, q_tensor)

    neighbor_ids, distances = index.search_knn(query_vectors=q, k=num_vectors)
    result = index.get_vector(neighbor_ids[0])
    assert np.allclose(a=q, b=result, atol=1e-5, rtol=1e-5)

    # Test consistency with manuallly created index
    new_index: SearchIndex = SearchIndex.create(
        metric_fn=metric_fn,
        num_dimensions=num_dimensions,
    )
    for x in space.vectors:
        new_index.add_item(vector=x)
    _assert_index_eq(index1=index, index2=new_index)

    new_index: SearchIndex = SearchIndex.create(
        metric_fn=metric_fn,
        num_dimensions=num_dimensions,
    )
    new_index.add_items(vectors=space.vectors)
    _assert_index_eq(index1=index, index2=new_index)

    # Test distance function
    # assert index.get_distance(x=space.vectors[0], y=space.vectors[1]) == new_index.get_distance(
    #     x=space.vectors[0], y=space.vectors[1]
    # )

    # Test serialization
    tmp_index = tmp_path / "index"
    index.save(filename=tmp_index)
    index_loaded: SearchIndex = SearchIndex.load(filename=tmp_index.with_suffix(".tar"))
    _assert_index_eq(index1=index, index2=index_loaded)

    assert index.metadata == index_loaded.metadata, (index.metadata["transform"], index_loaded.metadata["transform"])

    # Test efficient deletion
    # index_loaded.mark_deleted(id=0)
    # with pytest.raises(expected_exception=RuntimeError):
    #     index_loaded.mark_deleted(id=0)
    # with pytest.raises(expected_exception=RuntimeError):
    #     index_loaded.get_vector_by_key(key=str(0))
    # index_loaded.unmark_deleted(id=0)
    a = index_loaded.get_vector(offset=0)
    assert a is not None and len(a) == num_dimensions and np.allclose(a=a, b=index.get_vector(offset=0))

    # Test resize
    # index.max_elements
    # index.resize(new_size=3 * num_vectors)
    # assert index.max_elements == 3 * num_vectors

    # TODO: test indices with custom transform


@pytest.mark.parametrize("num_vectors", [100, 1_000])
@pytest.mark.parametrize("num_dimensions", [10, 256])
def test_transform(num_vectors: int, num_dimensions: int):
    seed_everything(seed=0)

    space = LatentSpace(
        vector_source=torch.randn(num_vectors, num_dimensions, dtype=torch.double),
        name="space1",
    )
    index1 = space.to_index(metric_fn=SearchMetric.COSINE)
    index2 = space.to_index(metric_fn=SearchMetric.INNER_PRODUCT, transform=lambda x: F.normalize(x, p=2, dim=1))

    for i in range(num_vectors):
        result1 = index1.search_knn(query_offsets=[i], k=3)
        result2 = index2.search_knn(query_offsets=[i], k=3)

        assert result1.offsets[0] == i
        assert result1.offsets[0] == result2.offsets[0]

    vectors = torch.randn(num_vectors, num_dimensions, dtype=torch.float32)

    space = LatentSpace(
        vector_source=vectors,
        name="space1",
    )
    index = space.to_index(metric_fn=SearchMetric.COSINE)
    assert index.transform is not None

    assert np.allclose(
        index.search_knn(query_vectors=vectors, k=10, transform=True).offsets,
        index.search_knn(query_vectors=F.normalize(vectors), k=10, transform=False).offsets,
    )
    assert np.allclose(
        index.search_knn(query_vectors=vectors, k=10, transform=True).offsets,
        index.search_knn(
            query_vectors=index.get_vectors(offsets=list(range(vectors.size(0)))), k=10, transform=False
        ).offsets,
    )


@pytest.mark.parametrize("num_vectors", [100, 1_000])
@pytest.mark.parametrize("num_dimensions", [10, 256, 1000])
@pytest.mark.parametrize(
    "search_metric2radius",
    [
        (SearchMetric.COSINE, 0.99),
        (SearchMetric.L2, 0.01),
        (SearchMetric.EUCLIDEAN, 0.01),
    ],
)
def test_range_search(num_vectors: int, num_dimensions: int, search_metric2radius: Tuple[SearchMetric, float]):
    search_metric, radius = search_metric2radius
    # if search_metric == SearchMetric.INNER_PRODUCT:
    #     pytest.skip("Range search not supported for inner product: it isn't a proper metric")

    seed_everything(seed=0)

    vectors = torch.randn(num_vectors, num_dimensions, dtype=torch.double) * 10
    space = LatentSpace(
        vector_source=vectors,
        name="space1",
    )

    index: SearchIndex = space.to_index(metric_fn=search_metric)
    for i in range(num_vectors):
        result = index.search_range(query_offsets=[i], radius=radius)
        vector_result = index.search_range(
            query_vectors=vectors[i], radius=radius, transform=search_metric.transformation
        )
        assert result.offsets[0] == i
        assert vector_result.offsets[0] == i


@pytest.mark.parametrize("num_vectors", [100, 500, 1_000])
def test_keys(
    num_vectors: int,
):
    seed_everything(seed=0)

    space = LatentSpace(
        vector_source=torch.randn(num_vectors, 100, dtype=torch.float32),
        name="space1",
    )

    keys = ["".join([chr(ord("a") + np.random.randint(0, 26)) for _ in range(8)]) for _ in range(num_vectors)]

    index: SearchIndex = space.to_index(metric_fn=SearchMetric.EUCLIDEAN, keys=keys)

    single_vector = torch.randn(100, dtype=torch.float32)
    index.add_item(vector=single_vector, key="single_additional_one")

    assert len(index) == num_vectors + 1
    assert index.num_elements == num_vectors + 1
    assert torch.allclose(single_vector, index.get_vector_by_key(key="single_additional_one", return_tensors=True))

    for i in range(num_vectors):
        result = index.search_knn(query_keys=[keys[i]], k=1, return_keys=True)
        (
            result_range_offsets,
            result_range_distances,
            result_range_keys,
        ) = index.search_range(query_keys=[keys[i]], radius=0.99, return_keys=True)
        assert result.offsets[0] == i
        assert result.keys[0] == keys[i]
        assert result_range_offsets[0] == i
        assert result_range_keys[0] == keys[i]


@pytest.mark.parametrize("num_vectors", [100, 500, 1_000])
@pytest.mark.parametrize("num_dimensions", [10, 256, 1000])
def test_get_vectors(num_vectors: int, num_dimensions: int):
    seed_everything(seed=0)

    index = SearchIndex.create(
        metric_fn=SearchMetric.EUCLIDEAN,
        num_dimensions=num_dimensions,
    )

    with pytest.raises(expected_exception=AssertionError):
        index.get_vector_by_key(key="impossible_key")

    vectors = torch.randn(num_vectors, num_dimensions, dtype=torch.float32)
    space = LatentSpace(
        vector_source=vectors,
        name="space1",
    )

    index = space.to_index(metric_fn=SearchMetric.EUCLIDEAN)
    retrieved_vectors = index.get_vectors(offsets=list(range(vectors.size(0))), return_tensors=True)

    assert torch.allclose(vectors, retrieved_vectors)
