import inspect
import re
from collections import UserDict, defaultdict
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import gin
import networkx as nx
from torch import nn

IOSpec = Union[str, Sequence[str], Mapping[str, str]]
Block = Union[nn.Module, Callable]

IOMappingRe = re.compile(r"(\w+):(\w+)")


class IOMapping(UserDict):
    def __init__(self, input_mappings: Mapping[str, Sequence[str]]):
        mappings = defaultdict(list)
        for mapping in input_mappings:
            if match := IOMappingRe.match(mapping):
                mappings[match.group(1)].append(match.group(2))
            else:
                mappings[mapping].append(mapping)
        super().__init__(mappings)

    def map(self, item: Union[str, tuple, Mapping[str, Any]]) -> Mapping[str, Any]:
        if isinstance(item, str):
            return {item: self[item]}
        elif isinstance(item, tuple):
            return {k: item[i] for i, k in enumerate(self)}
        elif isinstance(item, dict):
            return {v: item[k] for k, v in self.items()}
        else:
            raise ValueError(f"Unsupported type {type(item)}.")


class Step:
    def __init__(self, block: str, input_mappings: IOMapping, outputs: IOSpec, method: str = "__call__") -> None:
        self._input_mappings: IOMapping = input_mappings
        self._outputs: IOSpec = [outputs] if isinstance(outputs, str) else outputs
        self._method: str = method

        self._block: Block = block

    def __hash__(self) -> int:
        return hash((self._block, self._method, tuple(self._input_mappings), tuple(self._outputs)))

    @property
    def input_mappings(self) -> IOMapping:
        return self._input_mappings

    @property
    def outputs(self) -> IOSpec:
        return self._outputs

    @property
    def block(self) -> Optional[Block]:
        return self._block

    def __repr__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return self._block + (f".{self._method}" if self._method != "__call__" else "")

    def run(self, block: Block, **kwargs) -> Mapping[str, Any]:
        method = getattr(block, self._method)

        try:
            block_out = method(**kwargs)
        except TypeError as e:
            raise TypeError(f"Error when calling {block}.{self._method} with inputs {kwargs.keys()}.") from e

        # if isinstance(block_out, dict):
        #     raise NotImplementedError

        if not isinstance(block_out, tuple):
            # TODO: implement a better way to handle single outputs
            block_out = (block_out,)

        if len(self._outputs) != len(block_out):
            raise ValueError(
                f"Expected {len(self._outputs)} outputs, but got {len(block_out)}. Block: {block}. Method: {self._method}."
            )

        return dict(zip(self._outputs, block_out))


class FlowInput(Step):
    def __init__(self, name: str) -> None:
        super().__init__(input_mappings={name: [name]}, outputs=name, block=None, method=None)

    @property
    def name(self) -> str:
        return self._outputs[0]

    def run(self, block: Block, **kwargs) -> Mapping[str, Any]:
        return {self._outputs[0]: kwargs[self._outputs[0]]}


class FlowOutput(Step):
    def __init__(self, name: str) -> None:
        super().__init__(input_mappings={name: [name]}, outputs=name, block=None, method=None)

    @property
    def name(self) -> str:
        return self._outputs[0]

    def run(self, block: Block, **kwargs) -> Mapping[str, Any]:
        return {self._outputs[0]: kwargs[self._outputs[0]]}


class Passthrough(Step):
    def __init__(self, name: str) -> None:
        super().__init__(input_mappings={name: [name]}, outputs=name, block=None, method=None)

    @property
    def name(self) -> str:
        return self._outputs[0]

    def run(self, block: Block, **kwargs) -> Mapping[str, Any]:
        return {self._outputs[0]: kwargs[self._outputs[0]]}


class Flow:
    def __init__(self, name: str = "default", inputs: Sequence[str] = None, outputs: Sequence[str] = None) -> None:
        super().__init__()
        self.name: str = name
        inputs = [inputs] if isinstance(inputs, str) else inputs
        outputs = [outputs] if isinstance(outputs, str) else outputs

        self.dag = nx.MultiDiGraph()

        self.inputs: Optional[Sequence[FlowInput]] = None
        self.outputs = outputs

        # TODO: we could also have dynamic inputs and outputs... Do we prefer this fixed structure?
        if inputs is not None:
            self.inputs = [FlowInput(name=input_var) for input_var in inputs]
            for flow_input in self.inputs:
                self._add_step(step=flow_input)

        if outputs is not None:
            self.outputs = [FlowOutput(name=output_var) for output_var in outputs]
            for flow_output in self.outputs:
                self._add_step(step=flow_output)

    def to_pydot(self, **graph_attrs: Mapping[str, Any]):
        import graphviz

        graph = graphviz.Digraph(graph_attr=graph_attrs)

        node_padding = "4"
        edge_padding = "2"

        for node in self.dag.nodes:
            step = self.dag.nodes[node]["step"]
            if isinstance(step, FlowInput):
                node_shape = "square"
                color = "blue"
                # prefix: str = "in"
            elif isinstance(step, FlowOutput):
                node_shape = "square"
                color = "red"
                # prefix: str = "out"
            else:
                node_shape = "ellipse"
                color = "black"
                # prefix: str = ""

            node_label = f'<<table border="0" cellborder="0" cellspacing="0" cellpadding="{node_padding}"><tr><td>{step.name}</td></tr></table>>'
            graph.node(f"{node}", label=node_label, shape=node_shape, color=color)

        for edge in self.dag.edges:
            # target_step = self.dag.nodes[edge[1]]["step"]
            # if isinstance(target_step, FlowOutput):
            #     edge = list(edge)
            #     edge[1] = f"out{edge[1]}"
            #     edge = tuple(edge)
            edge_label = self.dag.edges[edge]["mappings"]
            edge_label = ", ".join(
                [k if len(v) == 1 and k == v[0] else f"{k}:{','.join(v)}" for k, v in edge_label.items()]
            )
            edge_label = f'<<table border="0" cellborder="0" cellspacing="0" cellpadding="{edge_padding}"><tr><td>{edge_label}</td></tr></table>>'
            graph.edge(edge[0], edge[1], label=edge_label)

        for step in self.get_leaves():
            step = self.dag.nodes[step.name]["step"]
            for output in step.outputs:
                node_label = f'<<table border="0" cellborder="0" cellspacing="0" cellpadding="{node_padding}"><tr><td>{output}</td></tr></table>>'
                graph.node(f"out{output}", shape="square", color="red", label=node_label)
                graph.edge(step.name, f"out{output}")

        return graph

    def get_flow_inputs(self) -> Sequence[Step]:
        return self.inputs
        # root_nodes = [node for node, in_degree in self.dag.in_degree if in_degree == 0]
        # return [self.dag.nodes[node]["step"] for node in root_nodes]

    def get_leaves(self) -> Sequence[Step]:
        leaf_nodes = [node for node, out_degree in self.dag.out_degree if out_degree == 0]
        return [self.dag.nodes[node]["step"] for node in leaf_nodes]

    def _add_step(self, step: Step) -> "Flow":
        self.dag.add_node(step.name, step=step)

    def get_by_output(self, output: str) -> Step:
        # traverse the graph in reverse topological order to find the nodes that provide the target output (excluding FlowOutputs)
        steps = [self.dag.nodes[node]["step"] for node in nx.topological_sort(self.dag.reverse())]
        steps = [step for step in steps if output in step.outputs and not isinstance(step, FlowOutput)]

        return steps

    def add(
        self,
        block: str,
        inputs: Sequence[IOSpec] = None,
        method: Optional[str] = "__call__",
        outputs: Sequence[IOSpec] = None,
    ) -> "Flow":
        if inputs is None:
            inputs = []

        if outputs is None:
            outputs = []

        if isinstance(inputs, str):
            inputs = [inputs]

        input_mappings = IOMapping(inputs)

        # check the block is a valid one
        # if block not in self.blocks:
        #     raise ValueError(f"Block name {block} not found in available blocks.")
        # block_obj = self.blocks[block]
        # Pipeline.check_signature(
        #     block=block_obj, method_name=method, inputs=list(input_mappings.values()), outputs=outputs
        # )

        step = Step(block=block, method=method, input_mappings=input_mappings, outputs=outputs)

        nodes = list(self.dag.nodes)

        # first, identify the step dependencies according to the input mappings
        dependencies: Mapping[str, Step] = {}
        for input_value, _ in step.input_mappings.items():
            # retrieve the UNIQUE leaf node that provides the input value excluding the FlowOutputs
            # if it's not found, then it's a new FlowInput
            possible_dependencies: Step = self.get_by_output(output=input_value)
            dependency = possible_dependencies[0] if len(possible_dependencies) > 0 else None

            if dependency is None:
                if self.inputs is not None:
                    raise ValueError(
                        f"Input {input_value} not found in predefined inputs: {self.inputs}. "
                        "Provide it or remove the predefined inputs from the constructor to dynamically infer them."
                    )
                else:  # dynamic flow input creation
                    dependency = FlowInput(name=input_value)
                    self._add_step(step=dependency)

            dependencies[input_value] = dependency

        # check the output of the step is not already provided by another step
        for output in step.outputs:
            for node in nodes:
                other_step: Step = self.dag.nodes[node]["step"]
                if isinstance(other_step, FlowOutput):
                    # remove previous incoming edges to the output node
                    self.dag.remove_edges_from(list(self.dag.in_edges(node)))
                    # add a new edge from the step to the output/leaf node
                    self.dag.add_edge(
                        step.name,
                        other_step.name,
                        mappings={}
                        # mappings={output: output},
                    )
                elif output in other_step.outputs and output not in dependencies.keys():
                    raise ValueError(f"Output {output} already provided by step {node}.")

        self._add_step(step=step)
        for input_value, dependency in dependencies.items():
            self.dag.add_edge(
                dependency.name,
                step.name,
                mappings=step.input_mappings.map(input_value),
            )

        # if not nx.is_directed_acyclic_graph(self.pipeline):
        #     raise RuntimeError("This flow isn't a DAG anymore!")

        return self

    def run(self, blocks: Mapping[str, Any], **kwargs):
        input_nodes = self.get_flow_inputs() or []
        if set(kwargs.keys()) != set(node.outputs[0] for node in input_nodes):
            raise ValueError(f"Expected inputs {input_nodes}, but got {kwargs.keys()}.")
        context = kwargs.copy()

        # traverse the graph in topological order
        for node in nx.topological_sort(self.dag):
            step: Step = self.dag.nodes[node]["step"]

            params = {
                target_name: context[source_name]
                for source_name, target_names in step.input_mappings.items()
                for target_name in target_names
            }
            try:
                block = blocks[step.block] if step.block is not None else None
            except KeyError as e:
                raise KeyError(f"Block {step.block} not found in available blocks: {blocks}") from e

            step_out = step.run(block=block, **params)

            context.update(step_out)

        return {flow_output.outputs[0]: context[flow_output.outputs[0]] for flow_output in self.outputs}


@gin.configurable("pipeline")
class Pipeline:
    def __init__(
        self,
        name: str,
        blocks: Optional[Mapping[str, Any]] = None,
        flows: Optional[Union[Flow, Mapping[str, Flow]]] = None,
    ) -> None:
        self.name = name
        self.blocks = blocks or {}
        flows = {flows.name: flows} if isinstance(flows, Flow) else flows
        self.flows: Mapping[str, Flow] = flows or {}

    # def flow(self, name: str) -> Flow:
    #     if name not in self.flows:
    #         self.flows[name] = Flow(name=name)
    #     return self.flows[name]

    def add(self, flow: Flow, force: bool = False) -> "Pipeline":
        if flow.name in self.flows and not force:
            raise ValueError(f"Flow {flow.name} already exists.")
        else:
            self.flows[flow.name] = flow

        return self

    def build(self, **blocks) -> "NNPipeline":
        # now the blocks shouldn't be placeholders anymore
        self.blocks.update(blocks)
        return self

    def run(self, flow: Optional[str] = None, **kwargs):
        if flow is None:
            if len(self.flows) != 1:
                raise ValueError("Multiple flows available. Please specify the flow to run.")
            else:
                flow = next(iter(self.flows.keys()))

        if flow is not None and flow not in self.flows:
            raise ValueError(f"Flow {flow} not found in available flows.")

        return self.flows[flow].run(blocks=self.blocks, **kwargs)

    def set(self, **params: Mapping[str, Any]) -> "Pipeline":
        for path, value in params.items():
            path = path.split(".")
            block_name = path[0]
            block = self.blocks[block_name]

            obj = block
            for attr in path[1:-1]:
                try:
                    obj = getattr(obj, attr)
                except AttributeError as e:
                    raise AttributeError(
                        f"Attribute {attr} not found in block {block_name}. The given path is {path}"
                    ) from e

            setattr(obj, path[-1], value)

        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    @classmethod
    def check_signature(
        cls, block: nn.Module, method_name: str, inputs: Sequence[str], outputs: Sequence[IOSpec]
    ) -> None:
        # check the method is a valid callable attribute of the block
        if isinstance(block, nn.Module) and isinstance(method_name, str):
            if not hasattr(block, method_name):
                raise ValueError(f"Method {method_name} not found in block {block}.")

        method = getattr(block, method_name)

        if not callable(method):
            raise ValueError(f"Method {method_name} is not callable.")

        # first, check the input signature
        input_args: Mapping[str] = inspect.signature(method).parameters
        # if an argument is given in inputs, then it should be mapped to an argument in the method signature
        # it can either be mapped directly or to a keyword argument
        kwargs = set(arg_name for arg_name, param in input_args.items() if param.kind == inspect.Parameter.VAR_KEYWORD)
        for input_name in inputs:
            if input_name not in input_args and len(kwargs) == 0:
                raise ValueError(f"Argument {input_name} not found in method {method_name} of {block} and no **kwargs.")

        # if an argument is required, it should be given in inputs
        # skip special arguments: self, cls, args, kwargs
        free_args = {
            arg_name: param
            for arg_name, param in input_args.items()
            if not (
                (arg_name in {"self", "cls"})
                or (param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD})
                or (arg_name not in inputs)
            )
        }
        for arg_name, param in free_args.items():
            if param.default == inspect.Parameter.empty and arg_name not in inputs:
                raise ValueError(f"Argument {arg_name} is required but not provided in inputs.")


class NNPipeline(nn.Module, Pipeline):
    def __init__(
        self, name: str, blocks: Optional[Mapping[str, Any]] = None, flows: Optional[Mapping[str, Flow]] = None
    ) -> None:
        nn.Module.__init__(self)
        Pipeline.__init__(self, name=name, blocks=blocks, flows=flows)

    def __repr__(self):
        return Pipeline.__repr__(self)

    def build(self, **blocks) -> "NNPipeline":
        # now the blocks shouldn't be placeholders anymore
        self.blocks.update(blocks)

        # add consistency checks

        for block_name, block in blocks.items():
            if not isinstance(block, nn.Module) and not callable(block):
                raise ValueError(f"Block name {block_name} is not a valid block.")

        self.blocks = nn.ModuleDict(self.blocks)

        return self
