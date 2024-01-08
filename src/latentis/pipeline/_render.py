import html
from typing import Iterable

import black
import networkx as nx

# import stackeddag.core as sd
from latentis.pipeline._utils import _is_passthrough

__all__ = ["DAGRenderer"]


class DAGRenderer:
    _EMPTY = "[empty]"
    _STYLES = {
        "light": {
            "node__color": "black",
            "node__fontcolor": "black",
            "edge__color": "black",
            "graph__bgcolor": "white",
        },
        "dark": {
            "node__color": "white",
            "node__fontcolor": "white",
            "edge__color": "white",
            "graph__bgcolor": "none",
        },
    }

    def __init__(self, dag, detailed=False, style=None):
        self.dag = dag
        if style is not None and style not in self._STYLES:
            raise ValueError(f"Unknown style: '{style}'.")
        self.style = style
        self.agraph = self.to_agraph(detailed)

    def _get_node_shape(self, estimator):
        if _is_passthrough(estimator):
            return "hexagon"
        if any(hasattr(estimator, attr) for attr in ["fit_predict", "predict"]):
            return "ellipse"
        return "box"

    def _is_empty(self):
        return len(self.dag) == 0

    def to_agraph(self, detailed):
        G = self.dag
        if self._is_empty():
            G = nx.DiGraph()
            G.add_node(
                self._EMPTY,
                shape="box",
            )

        try:
            A = nx.nx_agraph.to_agraph(G)
        except (ImportError, ModuleNotFoundError) as err:  # pragma: no cover
            raise ImportError(
                "DAG visualisation requires pygraphviz to be installed. "
                "See http://pygraphviz.github.io/ for guidance."
            ) from err

        A.graph_attr["rankdir"] = "LR"
        if self.style is not None:
            A.graph_attr.update(
                {
                    key.replace("graph__", ""): val
                    for key, val in self._STYLES[self.style].items()
                    if key.startswith("graph__")
                }
            )

        mode = black.FileMode()
        mode.line_length = 16

        for v in G.nodes:
            anode = A.get_node(v)
            gnode = G.nodes[v]
            anode.attr["fontname"] = gnode.get("fontname", "SANS")
            if self.style is not None:
                anode.attr.update(
                    {
                        key.replace("node__", ""): val
                        for key, val in self._STYLES[self.style].items()
                        if key.startswith("node__")
                    }
                )
            if "step" in gnode:
                estimator = gnode["step"].estimator
                anode.attr["tooltip"] = repr(estimator)

                if detailed:
                    estimator_str = html.escape(black.format_str(repr(estimator), mode=mode)).replace(
                        "\n", '<BR ALIGN="LEFT"/>'
                    )

                    anode.attr["label"] = (
                        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                        f'<TR><TD ALIGN="CENTER">{v}</TD></TR>'
                        f'<TR><TD ALIGN="LEFT"><FONT FACE="MONOSPACE" POINT-SIZE="6">{estimator_str}</FONT></TD></TR>'
                        "</TABLE>>"
                    )

                anode.attr["shape"] = gnode.get("shape", self._get_node_shape(estimator))
                if gnode["step"].is_fitted:
                    anode.attr["peripheries"] = 2

        for u, v in G.edges:
            aedge = A.get_edge(u, v)
            if self.style is not None:
                aedge.attr.update(
                    {
                        key.replace("edge__", ""): val
                        for key, val in self._STYLES[self.style].items()
                        if key.startswith("edge__")
                    }
                )
            cols = G.nodes[v]["step"].deps[u]
            if cols:
                if isinstance(cols, Iterable):
                    cols = list(cols)

                    if len(cols) > 5:
                        colrepr = f"[{repr(cols[0])}, ..., {repr(cols[-1])}]"
                    else:
                        colrepr = repr(cols)
                elif callable(cols):
                    selector = cols
                    cols = {}
                    for attr in ["pattern", "dtype_include", "dtype_exclude"]:
                        if hasattr(selector, attr):
                            val = getattr(selector, attr)
                            if val is not None:
                                cols[attr] = val
                    if cols:
                        selrepr = ", ".join(f"{key}={repr(val)}" for key, val in cols.items())
                        colrepr = f"column_selector({selrepr})"
                    else:
                        colrepr = f"{selector.__name__}()"
                else:
                    colrepr = repr(cols)

                aedge.attr.update({"label": colrepr, "fontsize": "8pt", "fontname": "SANS"})

        A.layout()
        return A

    def draw(self, format="svg", layout="dot"):
        A = self.agraph

        if format == "txt":
            if self._is_empty():
                return self._EMPTY

            # Edge case: single node, no edges.
            if len(A.edges()) == 0:
                return f"o    {next(A.nodes_iter())}"

            la = []
            for node in A.nodes():
                la.append((node, node))

            ed = []
            for src, dst in A.edges():
                ed.append((src, [dst]))

            return sd.edgesToText(sd.mkLabels(la), sd.mkEdges(ed)).strip()

        data = A.draw(format=format, prog=layout)

        if format == "svg":
            return data.decode(self.agraph.encoding)
        return data
