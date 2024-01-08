from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Sequence

from torch import Block, nn

# class Step(nn.Module):
#     def __init__(self, name: str) -> None:
#         super().__init__()
#         self.name: str = name


class Flow(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name: str = name
        self.blocks: Mapping[str, Block] = OrderedDict()


@dataclass
class FlowSpec:
    name: str
    inputs: Sequence[str]
    block_method: str

    def __post_init__(self):
        # TODO: check name validity
        pass

    def build(self) -> Flow:
        return Flow(name=self.name)


class Block(nn.Module):
    pass


if __name__ == "__main__":
    procrustes = (
        Pipeline(
            name="procrustes",
            # XtoYPipeline has those flows fixed
            flows=[
                FlowSpec(name="fit", inputs=["fit_x", "fit_y"], block_method="fit"),
                FlowSpec(name="transform", inputs=["x"], block_method="transform"),
            ],
        )
        .add("x_scaler", block=StandardScaler(), fit=["fit_x"], transform=["x"])
        .add("y_scaler", block=(y_scaler := StandardScaler()), fit="fit_y")
        .add("padding", block=ZeroPadding(), fit=["fit_x", "fit_y"], transform="x")
        .add(
            "estimator",
            block=SVDEstimator(),
            fit=[["fit_x", "fit_y"]],
            transform=[["x"], ["translated_x"]],
        )
        .add(y_scaler, transform_method="reverse", transform="translated_x")
    )

    aligner = (
        Pipeline(
            name="procrustes",
            # XtoYPipeline has those flows fixed
            flows=[
                FlowSpec(name="fit", inputs=["fit_x", "fit_y"], block_method="fit"),
                FlowSpec(name="transform", inputs=["x"], block_method="transform"),
            ],
        )
        .add("x_scaler", block=StandardScaler(), fit=["fit_x"], transform=["x"])
        .add("y_scaler", block=(y_scaler := StandardScaler()), fit="fit_y")
        .add("padding", block=ZeroPadding(), fit=["fit_x", "fit_y"], transform="x")
        .add(
            "estimator",
            fit=[["fit_x", "fit_y"]],
            transform=[["x"], ["translated_x"]],
        )
        .add(y_scaler, transform_method="reverse", transform="translated_x")
    )


if __name__ == "__main__":
    StandardScaler: int
    ZeroPadding: int
    Pipeline: int
    RelativeProjection: int
    Centering: int
    LinearAligner: int

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    procrustes = (
        Pipeline(fit=["fit_x", "fit_y"], transform=["x"])
        .add(x_scaler, fit_in="fit_x", fit_out="fit_x", transform="x")
        .add(y_scaler, fit="fit_y")
        .add(ZeroPadding(), fit=["fit_x", "fit_y"], transform="x")
        .add(
            SVDEstimator(),
            fit_in=["fit_x", "fit_y"],
            fit_out="translated_fit_x",
            transform_in="x",
            transform_out="translated_x",
        )
        .add(y_scaler.reverse, transform="translated_x")
    )

    relative_refined = (
        Pipeline(fit=["ref_x", "ref_y"], transform=["x"])
        .append(RelativeProjection(), fit=["ref_x"], transform_in=["x"], transform_out="rel_x")
        .append(RelativeProjection(), fit="ref_y")
        .append(Centering(), fit="ref_x", transform="rel_x")
        .append(Centering(), fit="ref_y")
        .append(LinearAligner(), fit_in=["ref_x", "ref_y"], transform="rel_x")
    )

    relative = Pipeline(fit=["ref_x"], transform=["x"]).append(
        RelativeProjection(), fit=["ref_x"], transform_in=["x"], transform_out="rel_x"
    )

    class XtoYPipeline(Pipeline):
        def fit(self, ref_x: torch.Tensor, ref_y: torch.Tensor):
            super().fit(ref_x, ref_y)

        def transform(self, x: torch.Tensor):
            super().transform(x)

    class XYtoZPipeline(Pipeline):
        def fit(self, ref_x: torch.Tensor, ref_y: torch.Tensor):
            super().fit(ref_x, ref_y)

        def transform(self, x: torch.Tensor, y: torch.Tensor):
            super().transform(x, y)

    class XPipeline(Pipeline):
        def fit(self, ref_x: torch.Tensor):
            super().fit(ref_x)

        def transform(self, x: torch.Tensor):
            super().transform(x)

    Aligner = (
        XtoYPipeline()
        .append(x_scaler, fit_in="fit_x", fit_out="fit_x", transform="x")
        .append(y_scaler, fit="fit_y")
        .append(ZeroPadding(), fit=["fit_x", "fit_y"], transform="x")
        .append(
            "estimator",
            fit_in=["fit_x", "fit_y"],
            fit_out="translated_fit_x",
            transform_in="x",
            transform_out="translated_x",
        )
        .append(y_scaler.reverse, transform="translated_x")
    ).build(estimator=SVDEstimator())

    Aligner.set(estimator=SVDEstimator()).build()
    Aligner.set(estimator=AffineEstimator()).build()

    RelativeProjection = XPipeline().append(RelativeProjection(), fit=["ref_x"], transform_in=["x"], transform_out="z")

    DynAnchorsRelativeProjection = (
        XPipeline()
        .append(Centering(), transform_in=["ref_x"], transform_out="ref_x")
        .append(RelativeProjection(), transform_in=["x", "ref_x"], transform_out="z")
    )

    CCAProjection = (
        XYtoZPipeline()
        .append(BohNormalizzazioniDiCCA(), fit="ref_x", transform_in=["x"])
        .append(BohNormalizzazioniDiCCA(), fit="ref_y", transform_in=["y"])
        .append(CCA(), fit=["ref_x", "ref_y"], transform=["x", "y"])
    )

if __name__ == "__main__":
    x = torch.randn(5, 2)
    anchors = x[torch.randperm(x.size(0))[:3]]

    RelRep = Pipeline(
        steps=[
            ("abs_transforms", "placeholder"),
            ("relative_projection", RelativeProjection(transform_fn=cosine_proj)),
            ("rel_transforms", "placeholder"),
        ]
    )

    RelRep.steps[0] = ("abs_transforms", Transform(transform_fn=centering))
    RelRep.steps[2] = ("rel_transforms", Transform(transform_fn=standard_scaling))
    relrep = RelRep.fit(anchors).transform(x)
    print(relrep)
    print(RelRep.set_params(**{"relative_projection__transform_fn": Transform(transform_fn=euclidean_proj)}))
    print(RelRep.transform(x))

    relrep1 = (
        Pipeline(
            steps=[
                ("centering", Transform(transform_fn=centering)),
                ("relative_projection", RelativeProjection(transform_fn=cosine_proj)),
            ]
        )
        .fit(anchors)
        .transform(x)
    )
    print(relrep1.shape)

    relrep2 = F.normalize(x - anchors.mean(dim=0)) @ F.normalize(anchors - anchors.mean(dim=0)).T

    print(torch.allclose(relrep1, relrep2))

    y = random_isometry(x, random_seed=42)
    y_scaling = StandardScaler()
    procrustes1 = (
        Pipeline(
            steps=[
                ("x_scaling", Transform(transform_fn=standard_scaling), "x"),
                ("y_scaling", y_scaling, "y"),
                ("padding", ZeroPadding(), ["x_scaling.0", "y_scaling"]),
                ("estimator", ["x", "y"], SVDEstimator()),
                ("y_descale", ["y"], Reverse(y_scaling)),
            ]
        )
        .fit(x=anchors_x, y=anchors_y)
        .transform(x)
    )
