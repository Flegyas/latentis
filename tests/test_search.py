from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from latentis.space import Space
from latentis.space.search import SearchMetric
from latentis.space.vector_source import SearchSource
from latentis.utils import seed_everything


def _assert_index_eq(space1: Space, space2: Space):
    # assert sorted(index1.keys) == sorted(index2.keys)
    assert len(space1) == len(space2)
    assert np.allclose(
        a=space1[range(len(space1))],
        b=space2[range(len(space1))],
    )
    # assert index1.key2offset == index2.key2offset
    assert np.allclose(a=space1.get_vector(offset=0), b=space2.get_vector(offset=0))


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

    space = Space(
        vector_source=(
            torch.randn(num_vectors, num_dimensions, dtype=torch.double),
            [str(i) for i in range(num_vectors)],
        ),
    )
    space = space.to_source(source_cls=SearchSource, metric_fn=metric_fn)

    # Test properties
    # assert sorted(index.ids) == sorted(range(num_vectors))
    # assert index.max_elements is not None
    num_elements, num_dimensions = space.shape()
    assert len(space) == num_vectors == num_elements
    assert space._vector_source._metric_fn == metric_fn
    # assert index.storage_data_type == DataType.FLOAT32
    assert str(space)

    space_keys = space.keys()
    for i in range(num_vectors):
        assert str(i) in space_keys

    # Test bytes serialization
    # assert index.as_bytes()

    # Test query
    q = space.get_vector(offset=0)
    assert torch.equal(torch.as_tensor(q), space.get_vector(offset=0))

    search_result1 = space.search_knn(query_vectors=q, k=num_vectors)
    search_result2 = space.search_knn(query_offsets=[0], k=num_vectors)
    assert np.allclose(search_result1.distances, search_result2.distances)
    assert np.allclose(search_result1.offsets, search_result2.offsets)

    dict_result = search_result1.asdict()
    assert isinstance(dict_result, dict)
    assert np.allclose(dict_result["offsets"], search_result1.offsets)
    assert np.allclose(dict_result["distances"], search_result1.distances)

    result = space.get_vector(offset=int(search_result1.offsets[0]))
    assert np.allclose(a=q, b=result, atol=1e-5, rtol=1e-5)
    assert len(search_result1.distances) == len(search_result1.offsets) == num_vectors

    # Test query with torch tensor
    q = torch.as_tensor(space.get_vector(offset=0))
    q_tensor = space.get_vector(offset=0)
    assert isinstance(q_tensor, torch.Tensor)
    assert np.allclose(q, q_tensor)

    search_result3 = space.search_knn(query_vectors=q, k=num_vectors)
    result = space.get_vector(search_result3.offsets[0])
    assert np.allclose(a=q, b=result, atol=1e-5, rtol=1e-5)

    # Test consistency with manuallly created index
    new_source: SearchSource = SearchSource.create(
        metric_fn=metric_fn,
        num_dimensions=num_dimensions,
    )
    for x in space.as_tensor():
        new_source.add_vector(vector=x)
    _assert_index_eq(space1=space, space2=Space(vector_source=new_source))

    new_source: SearchSource = SearchSource.create(
        metric_fn=metric_fn,
        num_dimensions=num_dimensions,
    )
    new_source.add_vectors(vectors=space.as_tensor())
    _assert_index_eq(space1=space, space2=Space(vector_source=new_source))

    # Test distance function
    # assert index.get_distance(x=space.vectors[0], y=space.vectors[1]) == new_index.get_distance(
    #     x=space.vectors[0], y=space.vectors[1]
    # )

    # Test serialization
    tmp_space_path = tmp_path / "index"
    space.save_to_disk(target_path=tmp_space_path)
    space_loaded: Space = Space.load_from_disk(path=tmp_space_path)
    _assert_index_eq(space1=space, space2=space_loaded)

    assert space.metadata == space_loaded.metadata, (space.metadata["transform"], space_loaded.metadata["transform"])

    # Test efficient deletion
    # index_loaded.mark_deleted(id=0)
    # with pytest.raises(expected_exception=RuntimeError):
    #     index_loaded.mark_deleted(id=0)
    # with pytest.raises(expected_exception=RuntimeError):
    #     index_loaded._get_vector_by_key(key=str(0))
    # index_loaded.unmark_deleted(id=0)
    a = space_loaded.get_vector(offset=0)
    assert a is not None and len(a) == num_dimensions and np.allclose(a=a, b=space.get_vector(offset=0))

    # Test resize
    # index.max_elements
    # index.resize(new_size=3 * num_vectors)
    # assert index.max_elements == 3 * num_vectors

    # TODO: test indices with custom transform


@pytest.mark.parametrize("num_vectors", [100, 1_000])
@pytest.mark.parametrize("num_dimensions", [10, 256])
def test_transform(num_vectors: int, num_dimensions: int):
    seed_everything(seed=0)

    space = Space(
        vector_source=torch.randn(num_vectors, num_dimensions, dtype=torch.double),
    )
    index1 = space.to_source(source_cls=SearchSource, metric_fn=SearchMetric.COSINE_SIM)
    index2 = space.to_source(
        source_cls=SearchSource, metric_fn=SearchMetric.INNER_PRODUCT, transform=lambda x: F.normalize(x, p=2, dim=1)
    )

    for i in range(num_vectors):
        result1 = index1.search_knn(query_offsets=[i], k=3)
        result2 = index2.search_knn(query_offsets=[i], k=3)

        assert result1.offsets[0] == i
        assert result1.offsets[0] == result2.offsets[0]

    vectors = torch.randn(num_vectors, num_dimensions, dtype=torch.float32)

    space = Space(
        vector_source=vectors,
    )
    space = space.to_source(source_cls=SearchSource, metric_fn=SearchMetric.COSINE_SIM)
    assert space.transform is not None

    assert np.allclose(
        space.search_knn(query_vectors=vectors, k=10, transform=True).offsets,
        space.search_knn(query_vectors=F.normalize(vectors), k=10, transform=False).offsets,
    )
    assert np.allclose(
        space.search_knn(query_vectors=vectors, k=10, transform=True).offsets,
        space.search_knn(query_vectors=space[range(vectors.size(0))], k=10, transform=False).offsets,
    )


@pytest.mark.parametrize("num_vectors", [100, 1_000])
@pytest.mark.parametrize("num_dimensions", [10, 256, 1000])
@pytest.mark.parametrize(
    "search_metric2radius",
    [
        (SearchMetric.COSINE_SIM, 0.99),
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
    space = Space(
        vector_source=vectors,
    )

    index: SearchSource = space.to_source(source_cls=SearchSource, metric_fn=search_metric)
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

    keys = ["".join([chr(ord("a") + np.random.randint(0, 26)) for _ in range(8)]) for _ in range(num_vectors)]

    space = Space(vector_source=(torch.randn(num_vectors, 100, dtype=torch.float32), keys))

    space: Space = space.to_source(source_cls=SearchSource, metric_fn=SearchMetric.EUCLIDEAN)

    single_vector = torch.randn(100, dtype=torch.float32)
    space.add_vectors(vectors=single_vector, keys=["single_additional_one"])

    assert len(space) == num_vectors + 1
    assert space.shape()[0] == num_vectors + 1
    assert np.allclose(single_vector, space.get_vector(key="single_additional_one"))

    for i in range(num_vectors):
        result = space.search_knn(query_keys=[keys[i]], k=1, return_keys=True)
        result_range = space.search_range(query_keys=[keys[i]], radius=0.99, return_keys=True)
        assert result.offsets[0] == i
        assert result.keys[0] == keys[i]
        assert result_range.offsets[0] == i
        assert result_range.keys[0] == keys[i], (result_range.keys[0], keys[i])


@pytest.mark.parametrize("num_vectors", [100, 500, 1_000])
@pytest.mark.parametrize("num_dimensions", [10, 256, 1000])
def test_get_vectors(num_vectors: int, num_dimensions: int):
    seed_everything(seed=0)

    index = SearchSource.create(
        metric_fn=SearchMetric.EUCLIDEAN,
        num_dimensions=num_dimensions,
    )

    with pytest.raises(expected_exception=AssertionError):
        index.get_vector(query_key="impossible_key")

    vectors = torch.randn(num_vectors, num_dimensions, dtype=torch.float32)
    space = Space(
        vector_source=(vectors, [str(i) for i in range(num_vectors)]),
    )

    index = space.to_source(source_cls=SearchSource, metric_fn=SearchMetric.EUCLIDEAN)
    retrieved_vectors = index[range(vectors.size(0))]

    assert np.allclose(vectors, retrieved_vectors)

    index = space.to_source(source_cls=SearchSource, metric_fn=SearchMetric.EUCLIDEAN)
    retrieved_vectors_by_keys = index.get_vectors_by_key(keys=[str(i) for i in range(num_vectors)])

    assert np.allclose(vectors, retrieved_vectors_by_keys)
