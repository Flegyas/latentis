import numpy as np
from scipy import sparse

try:
    import pandas as pd
except ImportError:
    pd = None


def _is_passthrough(estimator):
    return estimator is None or estimator == "passthrough"


def _is_transformer(estimator):
    return (hasattr(estimator, "fit") or hasattr(estimator, "fit_transform")) and hasattr(estimator, "transform")


def _is_predictor(estimator):
    return (hasattr(estimator, "fit") or hasattr(estimator, "fit_predict")) and hasattr(estimator, "predict")


def _in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def _stack(Xs, axis=0):
    """
    Where an estimator has multiple upstream dependencies, this method defines the
    strategy for merging the upstream outputs into a single input. ``axis=0`` is
    equivalent to a vstack (combination of samples) and ``axis=1`` is equivalent to a
    hstack (combination of features). Higher axes are only supported for non-sparse
    data sources.
    """
    if any(sparse.issparse(x) for x in Xs):
        if -2 <= axis < 2:
            axis = axis % 2
        else:
            raise NotImplementedError(f"Stacking is not supported for sparse inputs on axis > 1.")

        if axis == 0:
            Xs = sparse.vstack(Xs).tocsr()
        elif axis == 1:
            Xs = sparse.hstack(Xs).tocsr()
    elif pd and all(_is_pandas(x) for x in Xs):
        Xs = pd.concat(Xs, axis=axis)
    else:
        if axis == 1:
            Xs = np.hstack(Xs)
        else:
            Xs = np.stack(Xs, axis=axis)

    return Xs


def _is_pandas(X):
    "Check if X is a DataFrame or Series"
    return hasattr(X, "iloc")


def _get_feature_names(estimator):
    try:
        feature_names = estimator.get_feature_names_out()
    except AttributeError:
        try:
            feature_names = estimator.get_feature_names()
        except AttributeError:
            feature_names = None

    return feature_names


def _format_output(X, input, node):
    if node.dataframe_columns is None or pd is None or _is_pandas(X):
        return X

    outshape = np.asarray(X).shape
    outdim = len(outshape)
    if outdim > 2 or outdim < 1:
        return X

    if node.dataframe_columns == "infer":
        if outdim == 1:
            columns = [node.name]
        else:
            feature_names = _get_feature_names(node.estimator)
            if feature_names is None:
                feature_names = range(outshape[1])

            columns = [f"{node.name}__{f}" for f in feature_names]
    else:
        columns = node.dataframe_columns

    if hasattr(input, "index"):
        index = input.index
    else:
        index = None

    if outdim == 2:
        df = pd.DataFrame(X, columns=columns, index=index)
    else:
        df = pd.Series(X, name=columns[0], index=index)

    return df
