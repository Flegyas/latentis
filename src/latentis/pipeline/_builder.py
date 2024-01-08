from collections.abc import Mapping, Sequence

import networkx as nx

from latentis.pipeline.dagpipe import DAG, DAGStep

__all__ = ["DAGBuilder"]


class DAGBuilder:
    """
    Helper utility for creating a :class:`skdag.DAG`.

    ``DAGBuilder`` allows a graph to be defined incrementally by specifying one node
    (step) at a time. Graph edges are defined by providing optional dependency lists
    that reference each step by name. Note that steps must be defined before they are
    used as dependencies.

    Parameters
    ----------

    infer_dataframe : bool, default = False
        If True, assume ``dataframe_columns="infer"`` every time :meth:`.add_step` is
        called, if ``dataframe_columns`` is set to ``None``. This effectively makes the
        resulting DAG always try to coerce output into pandas DataFrames wherever
        possible.

    See Also
    --------
    :class:`skdag.DAG` : The estimator DAG created by this utility.

    Examples
    --------

    >>> from skdag import DAGBuilder
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.impute import SimpleImputer
    >>> from sklearn.linear_model import LogisticRegression
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
    """

    def __init__(self, infer_dataframe=False):
        self.graph = nx.DiGraph()
        self.infer_dataframe = infer_dataframe

    def from_pipeline(self, steps, **kwargs):
        """
        Construct a DAG from a simple linear sequence of steps. The resulting DAG will
        be equivalent to a :class:`~sklearn.pipeline.Pipeline`.

        Parameters
        ----------

        steps : sequence of (str, estimator)
            An ordered sequence of pipeline steps. A step is simply a pair of
            ``(name, estimator)``, just like a scikit-learn Pipeline.

        infer_dataframe : bool, default = False
            If True, assume ``dataframe_columns="infer"`` every time :meth:`.add_step`
            is called, if ``dataframe_columns`` is set to ``None``. This effectively
            makes the resulting DAG always try to coerce output into pandas DataFrames
            wherever possible.

        kwargs : kwargs
            Any other hyperparameters that are accepted by :class:`~skdag.dag.DAG`'s
            contructor.
        """
        if hasattr(steps, "steps"):
            pipe = steps
            steps = pipe.steps
            if hasattr(pipe, "get_params"):
                kwargs = {
                    **{k: v for k, v in pipe.get_params().items() if k in ("memory", "verbose")},
                    **kwargs,
                }

        dfcols = "infer" if self.infer_dataframe else None

        for i in range(len(steps)):
            name, estimator = steps[i]
            self._validate_name(name)
            deps = {}
            if i > 0:
                dep = steps[i - 1][0]
                deps[dep] = None
            self._validate_deps(deps)

            step = DAGStep(name, estimator, deps, dataframe_columns=dfcols)
            self.graph.add_node(name, step=step)
            if deps:
                self.graph.add_edge(dep, name)

        self._validate_graph()

        return self

    def add_step(self, name, est, deps=None, dataframe_columns=None):
        self._validate_name(name)
        if isinstance(deps, Sequence):
            deps = {dep: None for dep in deps}

        if deps is not None:
            self._validate_deps(deps)
        else:
            deps = {}

        if dataframe_columns is None and self.infer_dataframe:
            dfcols = "infer"
        else:
            dfcols = dataframe_columns

        step = DAGStep(name, est, deps=deps, dataframe_columns=dfcols)
        self.graph.add_node(name, step=step)

        for dep in deps:
            # Since node is new, edges will never form a cycle.
            self.graph.add_edge(dep, name)

        self._validate_graph()

        return self

    def _validate_name(self, name):
        if not isinstance(name, str):
            raise KeyError(f"step names must be strings, got '{type(name)}'")

        if name in self.graph.nodes:
            raise KeyError(f"step with name '{name}' already exists")

    def _validate_deps(self, deps):
        if not isinstance(deps, Mapping) or not all([isinstance(dep, str) for dep in deps]):
            raise ValueError("deps parameter must be a map of labels to indices, " f"got '{type(deps)}'.")

        missing = [dep for dep in deps if dep not in self.graph]
        if missing:
            raise ValueError(f"unresolvable dependencies: {', '.join(sorted(missing))}")

    def _validate_graph(self):
        if not nx.algorithms.dag.is_directed_acyclic_graph(self.graph):
            raise RuntimeError("Workflow is not a DAG.")

    def make_dag(self, **kwargs):
        self._validate_graph()
        # Give the DAG a read-only view of the graph.
        return DAG(graph=self.graph.copy(as_view=True), **kwargs)

    def _repr_html_(self):
        return self.make_dag()._repr_html_()
