# https://github.com/scikit-learn-contrib/skdag/blob/927805b69be29864682a16a9d425325f3db4a662/skdag/dag/_dag.py
"""
Directed Acyclic Graphs (DAGs) may be used to construct complex workflows for
scikit-learn estimators. As the name suggests, data may only flow in one
direction and can't go back on itself to a previously run step.
"""
from collections import UserDict
from copy import deepcopy
from inspect import signature
from typing import Iterable

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import dok_matrix, issparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils import Bunch, _print_elapsed_time, _safe_indexing, deprecated
from sklearn.utils._tags import _safe_tags
from sklearn.utils.metaestimators import _BaseComposition, available_if
from sklearn.utils.validation import check_is_fitted, check_memory

# from latentis.pipeline._render import DAGRenderer
from latentis.pipeline._utils import _format_output, _is_pandas, _is_passthrough, _is_predictor, _is_transformer, _stack

__all__ = ["DAG", "DAGStep"]


def _get_columns(X, dep, cols, is_root, dep_is_passthrough, axis=1):
    if callable(cols):
        # sklearn.compose.make_column_selector
        cols = cols(X)

    if not is_root and not dep_is_passthrough:
        # The DAG will prepend output columns with the step name, so add this in to any
        # dep columns if missing. This helps keep user-provided deps readable.
        if isinstance(cols, str):
            cols = cols if cols.startswith(f"{dep}__") else f"{dep}__{cols}"
        elif isinstance(cols, Iterable):
            orig = cols
            cols = []
            for col in orig:
                if isinstance(col, str):
                    cols.append(col if col.startswith(f"{dep}__") else f"{dep}__{col}")
                else:
                    cols.append(col)

    return _safe_indexing(X, cols, axis=axis)


def _stack_inputs(dag, X, node):
    # For root nodes, the dependency is just the node name itself.
    deps = {node.name: None} if node.is_root else node.deps

    cols = [
        _get_columns(
            X[dep],
            dep,
            cols,
            node.is_root,
            _is_passthrough(dag.graph_.nodes[dep]["step"].estimator),
            axis=1,
        )
        for dep, cols in deps.items()
    ]

    to_stack = [
        # If we sliced a single column from an input, reshape it to a 2d array.
        col.reshape(-1, 1) if col is not None and deps[dep] is not None and col.ndim < 2 else col
        for col, dep in zip(cols, deps)
    ]

    X_stacked = _stack(to_stack, axis=node.axis)

    return X_stacked


def _leaf_estimators_have(attr, how="all"):
    """Check that leaves have `attr`.
    Used together with `avaliable_if` in `DAG`."""

    def check_leaves(self):
        # raises `AttributeError` with all details if `attr` does not exist
        failed = []
        for leaf in self.leaves_:
            try:
                _is_passthrough(leaf.estimator) or getattr(leaf.estimator, attr)
            except AttributeError:
                failed.append(leaf.estimator)

        if (how == "all" and failed) or (how == "any" and len(failed) != len(self.leaves_)):
            raise AttributeError(
                f"{', '.join([repr(type(est)) for est in failed])} " f"object(s) has no attribute '{attr}'"
            )
        return True

    return check_leaves


def _transform_one(transformer, X, weight, allow_predictor=True, **fit_params):
    if _is_passthrough(transformer):
        res = X
    elif allow_predictor and not hasattr(transformer, "transform"):
        for fn in ["predict_proba", "decision_function", "predict"]:
            if hasattr(transformer, fn):
                res = getattr(transformer, fn)(X)
                if res.ndim < 2:
                    res = res.reshape(-1, 1)
                break
        else:
            raise AttributeError(f"'{type(transformer).__name__}' object has no attribute 'transform'")
    else:
        res = transformer.transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is not None:
        res = res * weight

    return res


def _fit_transform_one(
    transformer,
    X,
    y,
    weight,
    message_clsname="",
    message=None,
    allow_predictor=True,
    **fit_params,
):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    with _print_elapsed_time(message_clsname, message):
        failed = False
        if _is_passthrough(transformer):
            res = X
        elif hasattr(transformer, "fit_transform"):
            res = transformer.fit_transform(X, y, **fit_params)
        elif hasattr(transformer, "transform"):
            res = transformer.fit(X, y, **fit_params).transform(X)
        elif allow_predictor:
            for fn in ["predict_proba", "decision_function", "predict"]:
                if hasattr(transformer, fn):
                    res = getattr(transformer.fit(X, y, **fit_params), fn)(X)
                    if res.ndim < 2:
                        res = res.reshape(-1, 1)
                    break
            else:
                failed = True
                res = None

            if res is not None and res.ndim < 2:
                res = res.reshape(-1, 1)
        else:
            failed = True

        if failed:
            raise AttributeError(f"'{type(transformer).__name__}' object has no attribute 'transform'")

    if weight is not None:
        res = res * weight

    return res, transformer


def _parallel_fit(dag, step, Xin, Xs, y, fit_transform_fn, memory, **fit_params):
    transformer = step.estimator

    if step.deps:
        X = _stack_inputs(dag, Xs, step)
    else:
        # For root nodes, the destination rather than the source is
        # specified.
        # X = Xin[step.name]
        X = _stack_inputs(dag, Xin, step)

    clsname = type(dag).__name__
    with _print_elapsed_time(clsname, dag._log_message(step)):
        if transformer is None or transformer == "passthrough":
            Xt, fitted_transformer = X, transformer
        else:
            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)

            # Fit or load from cache the current transformer
            Xt, fitted_transformer = fit_transform_fn(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname=clsname,
                message=dag._log_message(step),
                **fit_params,
            )

    Xt = _format_output(Xt, X, step)

    return Xt, fitted_transformer


def _parallel_transform(dag, step, Xin, Xs, transform_fn, **fn_params):
    transformer = step.estimator
    if step.deps:
        X = _stack_inputs(dag, Xs, step)
    else:
        # For root nodes, the destination rather than the source is
        # specified.
        X = _stack_inputs(dag, Xin, step)
        # X = Xin[step.name]

    clsname = type(dag).__name__
    with _print_elapsed_time(clsname, dag._log_message(step)):
        if transformer is None or transformer == "passthrough":
            Xt = X
        else:
            # Fit or load from cache the current transformer
            Xt = transform_fn(
                transformer,
                X,
                None,
                message_clsname=clsname,
                message=dag._log_message(step),
                **fn_params,
            )

    Xt = _format_output(Xt, X, step)

    return Xt


def _parallel_fit_leaf(dag, leaf, Xts, y, **fit_params):
    with _print_elapsed_time(type(dag).__name__, dag._log_message(leaf)):
        if leaf.estimator == "passthrough":
            fitted_estimator = leaf.estimator
        else:
            Xt = _stack_inputs(dag, Xts, leaf)
            fitted_estimator = leaf.estimator.fit(Xt, y, **fit_params)

    return fitted_estimator


def _parallel_execute(dag, leaf, fn, Xts, y=None, fit_first=False, fit_params=None, fn_params=None):
    with _print_elapsed_time("DAG", dag._log_message(leaf)):
        Xt = _stack_inputs(dag, Xts, leaf)
        fit_params = fit_params or {}
        fn_params = fn_params or {}
        if leaf.estimator == "passthrough":
            Xout = Xt
        elif fit_first and hasattr(leaf.estimator, f"fit_{fn}"):
            Xout = getattr(leaf.estimator, f"fit_{fn}")(Xt, y, **fit_params)
        else:
            if fit_first:
                leaf.estimator.fit(Xt, y, **fit_params)

            est_fn = getattr(leaf.estimator, fn)
            if "y" in signature(est_fn).parameters:
                Xout = est_fn(Xt, y=y, **fn_params)
            else:
                Xout = est_fn(Xt, **fn_params)

        Xout = _format_output(Xout, Xt, leaf)

    fitted_estimator = leaf.estimator

    return Xout, fitted_estimator


class DAGStep:
    """
    A single estimator step in a DAG.

    Parameters
    ----------
    name : str
        The reference name for this step.
    estimator : estimator-like
        The estimator (transformer or predictor) that will be executed by this step.
    deps : dict
        A map of dependency names to columns. If columns is ``None``, then all input
        columns will be selected.
    dataframe_columns : list of str or "infer" (optional)
        Either a hard-coded list of column names to apply to any output data, or the
        string "infer", which means the column outputs will be assumed to match the
        column inputs if the output is 2d and not already a dataframe, the estimator is
        a transformer, and the final axis dimensions match the inputs. Otherwise the
        column names will be assumed to be the step name + index if the output is not
        already a dataframe. If set to ``None`` or inference is not possible, the
        outputs will be left unmodified.
    axis : int, default = 1
        The strategy for merging inputs if there is more than upstream dependency.
        ``axis=0`` will assume all inputs have the same features and stack the rows
        together; ``axis=1`` will assume each input provides different features for the
        same samples.
    """

    def __init__(self, name, estimator, deps, dataframe_columns, axis=1):
        self.name = name
        self.estimator = estimator
        self.deps = deps
        self.dataframe_columns = dataframe_columns
        self.axis = axis
        self.index = None
        self.is_root = False
        self.is_leaf = False
        self.is_fitted = False

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.name)}, {repr(self.estimator)})"


class DAG(_BaseComposition):
    """
    A Directed Acyclic Graph (DAG) of estimators, that itself implements the estimator
    interface.

    A DAG may consist of a simple chain of estimators (being exactly equivalent to a
    :mod:`sklearn.pipeline.Pipeline`) or a more complex path of dependencies. But as the
    name suggests, it may not contain any cyclic dependencies and data may only flow
    from one or more start points (roots) to one or more endpoints (leaves).

    Parameters
    ----------

    graph : :class:`networkx.DiGraph`
        A directed graph with string node IDs indicating the step name. Each node must
        have a ``step`` attribute, which contains a :class:`skdag.dag.DAGStep`.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the DAG. By default, no caching is
        performed. If a string is given, it is the path to the caching directory.
        Enabling caching triggers a clone of the transformers before fitting. Therefore,
        the transformer instance given to the DAG cannot be inspected directly. Use the
        attribute ``named_steps`` or ``steps`` to inspect estimators within the
        pipeline. Caching the transformers is advantageous when fitting is time
        consuming.

    n_jobs : int, default=None
        Number of jobs to run in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it is
        completed.

    Attributes
    ----------

    graph_ : :class:`networkx.DiGraph`
        A read-only view of the workflow.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. Only exists if the last step of the pipeline is a
        classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if all of the
        underlying root estimators in `graph_` expose such an attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the underlying
        estimators expose such an attribute when fit.

    See Also
    --------
    :class:`skdag.DAGBuilder` : Convenience utility for simplified DAG construction.

    Examples
    --------

    The simplest DAGs are just a chain of singular dependencies. These DAGs may be
    created from the :meth:`skdag.dag.DAG.from_pipeline` method in the same way as a
    DAG:

    >>> from sklearn.decomposition import PCA
    >>> from sklearn.impute import SimpleImputer
    >>> from sklearn.linear_model import LogisticRegression
    >>> dag = DAG.from_pipeline(
    ...     steps=[
    ...         ("impute", SimpleImputer()),
    ...         ("pca", PCA()),
    ...         ("lr", LogisticRegression())
    ...     ]
    ... )
    >>> print(dag.draw().strip())
    o    impute
    |
    o    pca
    |
    o    lr

    For more complex DAGs, it is recommended to use a :class:`skdag.dag.DAGBuilder`,
    which allows you to define the graph by specifying the dependencies of each new
    estimator:

    >>> from skdag import DAGBuilder
    >>> dag = (
    ...     DAGBuilder()
    ...     .add_step("impute", SimpleImputer())
    ...     .add_step("vitals", "passthrough", deps={"impute": slice(0, 4)})
    ...     .add_step("blood", PCA(n_components=2, random_state=0), deps={"impute": slice(4, 10)})
    ...     .add_step("lr", LogisticRegression(random_state=0), deps=["blood", "vitals"])
    ...     .make_dag()
    ... )
    >>> print(dag.draw().strip())
    o    impute
    |\\
    o o    blood,vitals
    |/
    o    lr

    In the above examples we pass the first four columns directly to a regressor, but
    the remaining columns have dimensionality reduction applied first before being
    passed to the same regressor. Note that we can define our graph edges in two
    different ways: as a dict (if we need to select only certain columns from the source
    node) or as a simple list (if we want to simply grab all columns from all input
    nodes).

    The DAG may now be used as an estimator in its own right:

    >>> from sklearn import datasets
    >>> X, y = datasets.load_diabetes(return_X_y=True)
    >>> dag.fit_predict(X, y)
    array([...

    In an extension to the scikit-learn estimator interface, DAGs also support multiple
    inputs and multiple outputs. Let's say we want to compare two different classifiers:

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> cal = DAG.from_pipeline(
    ...     [("rf", RandomForestClassifier(random_state=0))]
    ... )
    >>> dag2 = dag.join(cal, edges=[("blood", "rf"), ("vitals", "rf")])
    >>> print(dag2.draw().strip())
    o    impute
    |\\
    o o    blood,vitals
    |x|
    o o    lr,rf

    Now our DAG will return two outputs: one from each classifier. Multiple outputs are
    returned as a :class:`sklearn.utils.Bunch<Bunch>`:

    >>> y_pred = dag2.fit_predict(X, y)
    >>> y_pred.lr
    array([...
    >>> y_pred.rf
    array([...

    Similarly, multiple inputs are also acceptable and inputs can be provided by
    specifying ``X`` and ``y`` as a ``dict``-like object.
    """

    # BaseEstimator interface
    _required_parameters = ["graph"]

    @classmethod
    @deprecated(
        "DAG.from_pipeline is deprecated in 0.0.3 and will be removed in a future "
        "release. Please use DAGBuilder.from_pipeline instead."
    )
    def from_pipeline(cls, steps, **kwargs):
        from latentis.pipeline._builder import DAGBuilder

        return DAGBuilder().from_pipeline(steps, **kwargs).make_dag()

    def __init__(self, graph, *, memory=None, n_jobs=None, verbose=False):
        self.graph = graph
        self.memory = memory
        self.verbose = verbose
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        """
        Get parameters for this metaestimator.

        Returns the parameters given in the constructor as well as the
        estimators contained within the `steps_` of the `DAG`.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("steps_", deep=deep)

    def set_params(self, **params):
        """
        Set the parameters of this metaestimator.

        Valid parameter keys can be listed with ``get_params()``. Note that
        you can directly set the parameters of the estimators contained in
        `steps_`.

        Parameters
        ----------
        **params : dict
            Parameters of this metaestimator or parameters of estimators contained
            in `steps`. Parameters of the steps may be set using its name and
            the parameter name separated by a '__'.

        Returns
        -------
        self : object
            DAG class instance.
        """
        step_names = set(self.step_names)
        for param in list(params.keys()):
            if "__" not in param and param in step_names:
                self.graph_.nodes[param]["step"].estimator = params.pop(param)

        super().set_params(**params)
        return self

    def _log_message(self, step):
        if not self.verbose:
            return None

        return f"(step {step.name}: {step.index} of {len(self.graph_)}) Processing {step.name}"

    def _iter(self, with_leaves=True, filter_passthrough=True):
        """
        Generate stage lists from self.graph_.
        When filter_passthrough is True, 'passthrough' and None transformers
        are filtered out.
        """
        for stage in nx.topological_generations(self.graph_):
            stage = [self.graph_.nodes[step]["step"] for step in stage]
            if not with_leaves:
                stage = [step for step in stage if not step.is_leaf]

            if filter_passthrough:
                stage = [step for step in stage if step.estimator is not None and step.estimator != "passthough"]

            if len(stage) == 0:
                continue

            yield stage

    def __len__(self):
        """
        Returns the size of the DAG
        """
        return len(self.graph_)

    def __getitem__(self, name):
        """
        Retrieve a named estimator.
        """
        return self.graph_.nodes[name]["step"].estimator

    def _fit(self, X, y=None, **fit_params_steps):
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        root_names = set([root.name for root in self.roots_])
        Xin = self._resolve_inputs(X)
        Xs = {}
        with Parallel(n_jobs=self.n_jobs) as parallel:
            for stage in self._iter(with_leaves=False, filter_passthrough=False):
                stage_names = [step.name for step in stage]
                outputs, fitted_transformers = zip(
                    *parallel(
                        delayed(_parallel_fit)(
                            self,
                            step,
                            Xin,
                            Xs,
                            y,
                            fit_transform_one_cached,
                            memory,
                            **fit_params_steps[step.name],
                        )
                        for step in stage
                    )
                )

                for step, fitted_transformer in zip(stage, fitted_transformers):
                    # Replace the transformer of the step with the fitted
                    # transformer. This is necessary when loading the transformer
                    # from the cache.
                    step.estimator = fitted_transformer
                    step.is_fitted = True

                Xs.update(dict(zip(stage_names, outputs)))

                # If all of a dep's dependents are now complete, we can free up some
                # memory.
                root_names = root_names - set(stage_names)
                for dep in {dep for step in stage for dep in step.deps}:
                    dependents = self.graph_.successors(dep)
                    if all(d in Xs and d not in root_names for d in dependents):
                        del Xs[dep]

        # If a root node is also a leaf, it hasn't been fit yet and we need to pass on
        # its input for later.
        Xs.update({name: Xin[name] for name in root_names})
        return Xs

    def _transform(self, X, **fn_params_steps):
        # Setup the memory
        memory = check_memory(self.memory)

        transform_one_cached = memory.cache(_transform_one)

        root_names = set([root.name for root in self.roots_])
        Xin = self._resolve_inputs(X)
        Xs = {}
        with Parallel(n_jobs=self.n_jobs) as parallel:
            for stage in self._iter(with_leaves=False, filter_passthrough=False):
                stage_names = [step.name for step in stage]
                outputs = parallel(
                    delayed(_parallel_transform)(
                        self,
                        step,
                        Xin,
                        Xs,
                        transform_one_cached,
                        **fn_params_steps[step.name],
                    )
                    for step in stage
                )

                Xs.update(dict(zip(stage_names, outputs)))

                # If all of a dep's dependents are now complete, we can free up some
                # memory.
                root_names = root_names - set(stage_names)
                for dep in {dep for step in stage for dep in step.deps}:
                    dependents = self.graph_.successors(dep)
                    if all(d in Xs and d not in root_names for d in dependents):
                        del Xs[dep]

        # If a root node is also a leaf, it hasn't been fit yet and we need to pass on
        # its input for later.
        Xs.update({name: Xin[name] for name in root_names})
        return Xs

    def _resolve_inputs(self, X):
        if isinstance(X, (dict, Bunch, UserDict)) and not isinstance(X, dok_matrix):
            inputs = sorted(X.keys())
            if inputs != sorted(root.name for root in self.roots_):
                raise ValueError(
                    "Input dicts must contain one key per entry node. " f"Entry nodes are {self.roots_}, got {inputs}."
                )
        else:
            if len(self.roots_) != 1:
                raise ValueError("Must provide a dictionary of inputs for a DAG with multiple entry " "points.")
            X = {self.roots_[0].name: X}

        X = {step: x if issparse(x) or _is_pandas(x) else np.asarray(x) for step, x in X.items()}

        return X

    def _match_input_format(self, Xin, Xout):
        if len(self.leaves_) == 1 and (not isinstance(Xin, (dict, Bunch, UserDict)) or isinstance(Xin, dok_matrix)):
            return Xout[self.leaves_[0].name]
        return Bunch(**Xout)

    def fit(self, X, y=None, **fit_params):
        """
        Fit the model.

        Fit all the transformers one after the other and transform the
        data. Finally, fit the transformed data using the final estimators.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            DAG.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the DAG.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : object
            DAG fitted steps.
        """
        self._validate_graph()
        fit_params_steps = self._check_fit_params(**fit_params)
        Xts = self._fit(X, y, **fit_params_steps)
        fitted_estimators = Parallel(n_jobs=self.n_jobs)(
            [delayed(_parallel_fit_leaf)(self, leaf, Xts, y, **fit_params_steps[leaf.name]) for leaf in self.leaves_]
        )
        for est, leaf in zip(fitted_estimators, self.leaves_):
            leaf.estimator = est
            leaf.is_fitted = True

        # If we have a single root, mirror certain attributes in the DAG.
        if len(self.roots_) == 1:
            root = self.roots_[0].estimator
            for attr in ["n_features_in_", "feature_names_in_"]:
                if hasattr(root, attr):
                    setattr(self, attr, getattr(root, attr))

        return self

    def _fit_execute(self, fn, X, y=None, **fit_params):
        self._validate_graph()
        fit_params_steps = self._check_fit_params(**fit_params)
        Xts = self._fit(X, y, **fit_params_steps)
        Xout = {}

        leaf_names = [leaf.name for leaf in self.leaves_]
        outputs, fitted_estimators = zip(
            *Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_execute)(
                    self,
                    leaf,
                    fn,
                    Xts,
                    y,
                    fit_first=True,
                    fit_params=fit_params_steps[leaf.name],
                )
                for leaf in self.leaves_
            )
        )

        Xout = dict(zip(leaf_names, outputs))
        for step, fitted_estimator in zip(self.leaves_, fitted_estimators):
            step.estimator = fitted_estimator
            step.is_fitted = True

        return self._match_input_format(X, Xout)

    def _execute(self, fn, X, y=None, **fn_params):
        Xout = {}
        fn_params_steps = self._check_fn_params(**fn_params)
        Xts = self._transform(X, **fn_params_steps)

        leaf_names = [leaf.name for leaf in self.leaves_]
        outputs, _ = zip(
            *Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_execute)(
                    self,
                    leaf,
                    fn,
                    Xts,
                    y,
                    fit_first=False,
                    fn_params=fn_params_steps[leaf.name],
                )
                for leaf in self.leaves_
            )
        )

        Xout = dict(zip(leaf_names, outputs))

        return self._match_input_format(X, Xout)

    @available_if(_leaf_estimators_have("transform"))
    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit the model and transform with the final estimator.

        Fits all the transformers one after the other and transform the
        data. Then uses `fit_transform` on transformed data with the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            DAG.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the DAG.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        return self._fit_execute("transform", X, y, **fit_params)

    @available_if(_leaf_estimators_have("transform"))
    def transform(self, X):
        """
        Transform the data, and apply `transform` with the final estimator.

        Call `transform` of each transformer in the DAG. The transformed
        data are finally passed to the final estimator that calls
        `transform` method. Only valid if the final estimator
        implements `transform`.

        This also works where final estimator is `None` in which case all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the DAG.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        return self._execute("transform", X)

    @available_if(_leaf_estimators_have("predict"))
    def fit_predict(self, X, y=None, **fit_params):
        """
        Transform the data, and apply `fit_predict` with the final estimator.

        Call `fit_transform` of each transformer in the DAG. The transformed data are
        finally passed to the final estimator that calls `fit_predict` method. Only
        valid if the final estimators implement `fit_predict`.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the DAG.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the DAG.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : ndarray
            Result of calling `fit_predict` on the final estimator.
        """
        return self._fit_execute("predict", X, y, **fit_params)

    @available_if(_leaf_estimators_have("predict"))
    def predict(self, X, **predict_params):
        """
        Transform the data, and apply `predict` with the final estimator.

        Call `transform` of each transformer in the DAG. The transformed
        data are finally passed to the final estimator that calls `predict`
        method. Only valid if the final estimators implement `predict`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the DAG.
        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the DAG. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the DAG are not propagated to the
            final estimator.

        Returns
        -------
        y_pred : ndarray
            Result of calling `predict` on the final estimator.
        """
        return self._execute("predict", X, **predict_params)

    @available_if(_leaf_estimators_have("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        """
        Transform the data, and apply `predict_proba` with the final estimator.

        Call `transform` of each transformer in the DAG. The transformed
        data are finally passed to the final estimator that calls
        `predict_proba` method. Only valid if the final estimators implement
        `predict_proba`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the DAG.
        **predict_proba_params : dict of string -> object
            Parameters to the `predict_proba` called at the end of all
            transformations in the DAG.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_proba` on the final estimator.
        """
        return self._execute("predict_proba", X, **predict_proba_params)

    @available_if(_leaf_estimators_have("decision_function"))
    def decision_function(self, X):
        """
        Transform the data, and apply `decision_function` with the final estimator.

        Call `transform` of each transformer in the DAG. The transformed
        data are finally passed to the final estimator that calls
        `decision_function` method. Only valid if the final estimators
        implement `decision_function`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the DAG.

        Returns
        -------
        y_score : ndarray of shape (n_samples, n_classes)
            Result of calling `decision_function` on the final estimator.
        """
        return self._execute("decision_function", X)

    @available_if(_leaf_estimators_have("score_samples"))
    def score_samples(self, X):
        """
        Transform the data, and apply `score_samples` with the final estimator.

        Call `transform` of each transformer in the DAG. The transformed
        data are finally passed to the final estimator that calls
        `score_samples` method. Only valid if the final estimators implement
        `score_samples`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the DAG.

        Returns
        -------
        y_score : ndarray of shape (n_samples,)
            Result of calling `score_samples` on the final estimator.
        """
        return self._execute("score_samples", X)

    @available_if(_leaf_estimators_have("score"))
    def score(self, X, y=None, **score_params):
        """
        Transform the data, and apply `score` with the final estimator.

        Call `transform` of each transformer in the DAG. The transformed
        data are finally passed to the final estimator that calls
        `score` method. Only valid if the final estimators implement `score`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the DAG.
        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the DAG.
        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        Returns
        -------
        score : float
            Result of calling `score` on the final estimator.
        """
        return self._execute("score", X, y, **score_params)

    @available_if(_leaf_estimators_have("predict_log_proba"))
    def predict_log_proba(self, X, **predict_log_proba_params):
        """
        Transform the data, and apply `predict_log_proba` with the final estimator.

        Call `transform` of each transformer in the DAG. The transformed
        data are finally passed to the final estimator that calls
        `predict_log_proba` method. Only valid if the final estimator
        implements `predict_log_proba`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the DAG.
        **predict_log_proba_params : dict of string -> object
            Parameters to the ``predict_log_proba`` called at the end of all
            transformations in the DAG.

        Returns
        -------
        y_log_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_log_proba` on the final estimator.
        """
        return self._execute("predict_log_proba", X, **predict_log_proba_params)

    def _check_fit_params(self, **fit_params):
        fit_params_steps = {name: {} for (name, step) in self.steps_ if step is not None}
        for pname, pval in fit_params.items():
            if pval is None:
                continue

            if "__" not in pname:
                raise ValueError(
                    f"DAG.fit does not accept the {pname} parameter. "
                    "You can pass parameters to specific steps of your "
                    "DAG using the stepname__parameter format, e.g. "
                    "`DAG.fit(X, y, logisticregression__sample_weight"
                    "=sample_weight)`."
                )
            step, param = pname.split("__", 1)
            fit_params_steps[step][param] = pval
        return fit_params_steps

    def _check_fn_params(self, **fn_params):
        global_params = {}
        fn_params_steps = {name: {} for (name, step) in self.steps_ if step is not None}
        for pname, pval in fn_params.items():
            if pval is None:
                continue

            if "__" not in pname:
                global_params[pname] = pval
            else:
                step, param = pname.split("__", 1)
                fn_params_steps[step][param] = pval

        for step in fn_params_steps:
            fn_params_steps[step].update(global_params)

        return fn_params_steps

    def _validate_graph(self):
        if len(self.graph_) == 0:
            raise ValueError("DAG has no nodes.")

        for i, (name, est) in enumerate(self.steps_):
            step = self.graph_.nodes[name]["step"]
            step.index = i

        # validate names
        self._validate_names([name for (name, step) in self.steps_])

        # validate transformers
        for step in self.roots_ + self.branches_:
            if step in self.leaves_:
                # This will get validated later
                continue

            est = step.estimator
            # Unlike pipelines we also allow predictors to be used as a transformer, to support
            # model stacking.
            if not _is_passthrough(est) and not _is_transformer(est) and not _is_predictor(est):
                raise TypeError(
                    "All intermediate steps should be "
                    "transformers and implement fit and transform "
                    "or be the string 'passthrough' "
                    f"'{est}' (type {type(est)}) doesn't"
                )

        # Validate final estimator(s)
        for step in self.leaves_:
            est = step.estimator
            if not _is_passthrough(est) and not hasattr(est, "fit"):
                raise TypeError(
                    "Leaf nodes of a DAG should implement fit "
                    "or be the string 'passthrough'. "
                    f"'{est}' (type {type(est)}) doesn't"
                )

    @property
    def graph_(self):
        if not hasattr(self, "_graph"):
            # Read-only view of the graph. We should not modify
            # the original graph.
            self._graph = self.graph.copy(as_view=True)

        return self._graph

    @property
    def leaves_(self):
        if not hasattr(self, "_leaves"):
            self._leaves = [node for node in self.nodes_ if node.is_leaf]

        return self._leaves

    @property
    def branches_(self):
        if not hasattr(self, "_branches"):
            self._branches = [node for node in self.nodes_ if not node.is_leaf and not node.is_root]

        return self._branches

    @property
    def roots_(self):
        if not hasattr(self, "_roots"):
            self._roots = [node for node in self.nodes_ if node.is_root]

        return self._roots

    @property
    def nodes_(self):
        if not hasattr(self, "_nodes"):
            self._nodes = []
            for name, estimator in self.steps_:
                step = self.graph_.nodes[name]["step"]
                if self.graph_.out_degree(name) == 0:
                    step.is_leaf = True
                if self.graph_.in_degree(name) == 0:
                    step.is_root = True
                self._nodes.append(step)

        return self._nodes

    @property
    def steps_(self):
        "return list of (name, estimator) tuples to conform with Pipeline interface."
        if not hasattr(self, "_steps"):
            self._steps = [
                (node, self.graph_.nodes[node]["step"].estimator)
                for node in nx.lexicographical_topological_sort(self.graph_)
            ]

        return self._steps

    def join(self, other, edges, **kwargs):
        """
        Create a new DAG by joining this DAG to another one, according to the edges
        specified.

        Parameters
        ----------

        other : :class:`skdag.dag.DAG`
            The other DAG to connect to.
        edges : (str, str) or (str, str, index-like)
            ``(u, v)`` edges that connect the two DAGs. ``u`` and ``v`` should be the
            names of steps in the first and second DAG respectively. Optionally a third
            parameter may be included to specify which columns to pass along the edge.
        **kwargs : keyword params
            Any other parameters to pass to the new DAG's constructor.

        Returns
        -------
        dag : :class:`skdag.DAG`
            A new DAG, containing a copy of each of the input DAGs, joined by the
            specified edges. Note that the original input dags are unmodified.

        Examples
        --------

        >>> from sklearn.decomposition import PCA
        >>> from sklearn.impute import SimpleImputer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.calibration import CalibratedClassifierCV
        >>> from skdag.dag import DAGBuilder
        >>> dag1 = (
        ...     DAGBuilder()
        ...     .add_step("impute", SimpleImputer())
        ...     .add_step("vitals", "passthrough", deps={"impute": slice(0, 4)})
        ...     .add_step("blood", PCA(n_components=2, random_state=0), deps={"impute": slice(4, 10)})
        ...     .add_step("lr", LogisticRegression(random_state=0), deps=["blood", "vitals"])
        ...     .make_dag()
        ... )
        >>> print(dag1.draw().strip())
        o    impute
        |\\
        o o    blood,vitals
        |/
        o    lr
        >>> dag2 = (
        ...     DAGBuilder()
        ...     .add_step(
        ...         "calib",
        ...         CalibratedClassifierCV(LogisticRegression(random_state=0), cv=5),
        ...     )
        ...     .make_dag()
        ... )
        >>> print(dag2.draw().strip())
        o    calib
        >>> dag3 = dag1.join(dag2, edges=[("blood", "calib"), ("vitals", "calib")])
        >>> print(dag3.draw().strip())
        o    impute
        |\\
        o o    blood,vitals
        |x|
        o o    calib,lr
        """
        if set(self.step_names) & set(other.step_names):
            raise ValueError("DAGs with overlapping step names cannot be combined.")

        newgraph = deepcopy(self.graph_).copy()
        for edge in edges:
            if len(edge) == 2:
                u, v, idx = *edge, None
            else:
                u, v, idx = edge

            if u not in self.graph_:
                raise KeyError(u)
            if v not in other.graph_:
                raise KeyError(v)

            # source node can no longer be a leaf
            ustep = newgraph.nodes[u]["step"]
            if ustep.is_leaf:
                ustep.is_leaf = False

            vnode = other.graph_.nodes[v]
            old_step = vnode["step"]
            vstep = DAGStep(
                name=old_step.name,
                estimator=old_step.estimator,
                deps=old_step.deps,
                dataframe_columns=old_step.dataframe_columns,
                axis=old_step.axis,
            )

            if u not in vstep.deps:
                vstep.deps[u] = idx

            vnode["step"] = vstep

            newgraph.add_node(v, **vnode)
            newgraph.add_edge(u, v)

        return DAG(newgraph, **kwargs)

    # def draw(self, filename=None, style=None, detailed=False, format=None, layout="dot"):
    #     """
    #     Render a graphical view of the DAG.

    #     By default the rendered file will be returned as a string. However if an output
    #     file is provided then the output will be saved to file.

    #     Parameters
    #     ----------

    #     filename : str
    #         The file to write the image to. If None, the rendered image will be sent to
    #         stdout.
    #     style : str, optional, choice of ['light', 'dark']
    #         Draw the image in light or dark mode.
    #     detailed : bool, default = False
    #         If True, show extra details in the node labels such as the estimator
    #         signature.
    #     format : str, choice of ['svg', 'png', 'jpg', 'txt']
    #         The rendering format to use. MAy be omitted if the format can be inferred
    #         from the filename.
    #     layout : str, default = 'dot'
    #         The program to use for generating a graph layout.

    #     See Also
    #     --------

    #     :meth:`skdag.dag.DAG.show`, for use in interactive notebooks.

    #     Returns
    #     -------

    #     output : str, bytes or None
    #         If a filename is provided the output is written to file and `None` is
    #         returned. Otherwise, the output is returned as a string (for textual formats
    #         like ascii or svg) or bytes.
    #     """
    #     if filename is None and format is None:
    #         try:
    #             from IPython import get_ipython

    #             rich = type(get_ipython()).__name__ == "ZMQInteractiveShell"
    #         except (ModuleNotFoundError, NameError):
    #             rich = False

    #         format = "svg" if rich else "txt"

    #     if format is None:
    #         format = filename.split(".")[-1]

    #     if format not in ["svg", "png", "jpg", "txt"]:
    #         raise ValueError(f"Unsupported file format '{format}'")

    #     render = DAGRenderer(self.graph_, detailed=detailed, style=style).draw(format=format, layout=layout)
    #     if filename is None:
    #         return render
    #     else:
    #         mode = "wb" if isinstance(render, bytes) else "w"
    #         with open(filename, mode) as fp:
    #             fp.write(render)

    # def show(self, style=None, detailed=False, format=None, layout="dot"):
    #     """
    #     Display a graphical representation of the DAG in an interactive notebook
    #     environment.

    #     DAGs will be shown when displayed in a notebook, but calling this method
    #     directly allows more options to be passed to customise the appearance more.

    #     Arguments are as for :meth`.draw`.

    #     Returns
    #     -------

    #     ``None``

    #     See Also
    #     --------

    #     :meth:`skdag.DAG.draw`
    #     """
    #     if format is None:
    #         format = "svg" if _in_notebook() else "txt"

    #     data = self.draw(style=style, detailed=detailed, format=format, layout=layout)
    #     if format == "svg":
    #         from IPython.display import SVG, display

    #         display(SVG(data))
    #     elif format == "txt":
    #         print(data)
    #     elif format in ("jpg", "png"):
    #         from IPython.display import Image, display

    #         display(Image(data))
    #     else:
    #         raise ValueError(f"'{format}' format not supported.")

    # def _repr_svg_(self):
    #     return self.draw(format="svg")

    # def _repr_png_(self):
    #     return self.draw(format="png")

    # def _repr_jpeg_(self):
    #     return self.draw(format="jpg")

    # def _repr_html_(self):
    #     return self.draw(format="svg")

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(repr(self))
        else:
            p.text(str(self))

    def _repr_mimebundle_(self, **kwargs):
        # Don't render yet...
        renderers = {
            "image/svg+xml": self._repr_svg_,
            "image/png": self._repr_png_,
            "image/jpeg": self._repr_jpeg_,
            "text/plain": self.__str__,
            "text/html": self._repr_html_,
        }

        include = kwargs.get("include")
        if include:
            renderers = {k: v for k, v in renderers.items() if k in include}

        exclude = kwargs.get("exclude")
        if exclude:
            renderers = {k: v for k, v in renderers.items() if k not in exclude}

        # Now render any remaining options.
        return {k: v() for k, v in renderers.items()}

    @property
    def named_steps(self):
        """
        Access the steps by name.

        Read-only attribute to access any step by given name.
        Keys are steps names and values are the steps objects.
        """
        # Use Bunch object to improve autocomplete
        return Bunch(**dict(self.steps_))

    @property
    def step_names(self):
        return list(self.graph_.nodes)

    @property
    def edges(self):
        return self.graph_.edges

    def _get_leaf_attr(self, attr):
        if len(self.leaves_) == 1:
            return getattr(self.leaves_[0].estimator, attr)
        else:
            return Bunch(**{leaf.name: getattr(leaf.estimator, attr) for leaf in self.leaves_})

    @property
    def _estimator_type(self):
        return self._get_leaf_attr("_estimator_type")

    @property
    def classes_(self):
        """The classes labels. Only exist if the leaf steps are classifiers."""
        return self._get_leaf_attr("classes_")

    def __sklearn_is_fitted__(self):
        """Indicate whether DAG has been fit."""
        try:
            # check if the last steps of the DAG are fitted
            # we only check the last steps since if the last steps are fit, it
            # means the previous steps should also be fit. This is faster than
            # checking if every step of the DAG is fit.
            for leaf in self._leaves:
                check_is_fitted(leaf.estimator)
            return True
        except NotFittedError:
            return False

    def _more_tags(self):
        tags = {}

        # We assume the DAG can handle NaN if *all* the steps can.
        tags["allow_nan"] = all(_safe_tags(node.estimator, "allow_nan") for node in self.nodes_)

        # Check if all *root* nodes expect pairwise input.
        tags["pairwise"] = all(_safe_tags(root.estimator, "pairwise") for root in self.roots_)

        # CHeck if all *leaf* notes support multioutput
        tags["multioutput"] = all(_safe_tags(leaf.estimator, "multioutput") for leaf in self.leaves_)

        return tags


if __name__ == "__main__":
    print(dag)
