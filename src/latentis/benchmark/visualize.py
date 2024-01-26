import networkx as nx
from bokeh.io import show
from bokeh.models import Circle, HoverTool, MultiLine
from bokeh.plotting import figure, from_networkx
from networkx.drawing.nx_agraph import graphviz_layout

from latentis.benchmark.compiler import compile_experiments

LINE_ALPHA = 0.5
LINE_WIDTH = 0.75

ON_HOVER_ORDER = [
    "stage",
    "estimator",
    "correspondence_fit_id",
    "space_x_fit_id",
    "space_y_fit_id",
    "correspondence_test_id",
    "space_x_test_id",
    "space_y_test_id",
    "decoder_y_fit_id",
    "y_gt_test",
    "metric_id",
]

STAGE_2_COLORSIZE = {
    "estimation": {"color": "#F28585", "size": 22.5},
    "transform": {"color": "#FFA447", "size": 15},
    "latent": {"color": "#7BD3EA", "size": 9},
    "downstream": {"color": "#F1F7B5", "size": 9},
    "agreement": {"color": "#006837", "size": 9},
}


def _on_hover_str_format(x: str) -> str:
    return x[:8]


def _get_serializable_graph(graph: nx.Graph) -> nx.Graph:
    # Create a serializable graph
    Gs = nx.Graph()
    Gs.add_edges_from(graph.edges())

    # Add job attributes to node with short names
    for key, node_data in graph.nodes.data():
        for name_attr, attr in node_data["job"].node_info().items():
            Gs.nodes[key][name_attr] = _on_hover_str_format(attr[:8])

    return Gs


def display_benchmark_graph(benchmark_name: str) -> None:
    graph, result = compile_experiments(benchmark_name)

    stage2nodes = {
        "estimation": result["estimations"],
        "transform": result["transformations"],
        "latent": result["latents"],
        "downstream": result["downstreams"],
        "agreement": result["agreements"],
    }

    p = figure(
        tools="pan,wheel_zoom,save,reset",
        active_scroll="wheel_zoom",
        title=f"Benchmark: {benchmark_name}",
    )
    p.sizing_mode = "stretch_both"

    # TODO: on hover should show only stage-specific information
    all_infos = set([key for _, node_data in graph.nodes.data() for key in node_data["job"].node_info().keys()])
    all_infos.add("stage")
    all_infos = sorted(all_infos, key=lambda x: ON_HOVER_ORDER.index(x) if x in ON_HOVER_ORDER else 999)
    p.add_tools(
        HoverTool(tooltips=[(info, f"@{info}") for info in all_infos]),
    )

    graph_s = _get_serializable_graph(graph)

    for stage_name in ["estimation", "transform", "latent", "downstream", "agreement"]:
        for node_id in stage2nodes[stage_name]:
            graph_s.nodes[node_id]["stage"] = stage_name
            for key in ["color", "size"]:
                graph_s.nodes[node_id][key] = STAGE_2_COLORSIZE[stage_name][key]

    # hach to add the legend
    for k, v in STAGE_2_COLORSIZE.items():
        p.circle(x=0, y=0, size=0, legend_label=k, color=v["color"])

    graph_bokeh = from_networkx(graph_s, graphviz_layout(graph_s, prog="twopi"))

    graph_bokeh.node_renderer.glyph = Circle(fill_color="color", size="size")
    graph_bokeh.edge_renderer.glyph = MultiLine(line_alpha=LINE_ALPHA, line_width=LINE_WIDTH)

    p.renderers.append(graph_bokeh)
    p.axis.visible = False

    show(p)


if __name__ == "__main__":
    display_benchmark_graph("eval_estimator")
