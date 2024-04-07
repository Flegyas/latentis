import itertools
from typing import Callable, Dict, Sequence

import torch
from torch import nn

from latentis.data import DATA_DIR
from latentis.data.dataset import DatasetView
from latentis.nexus import space_index
from latentis.nn import LatentisModule
from latentis.nn.decoders import Classifier


def fit_model(
    model: LatentisModule,
    dataset: DatasetView,
    train_space_id: str,
    test_space_id: str,
    y_gt_key: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    **trainer_params,
):
    model = model.to(device)

    train_dataloader = dataset.get_dataloader(
        space_id=train_space_id,
        hf_x_keys=None,
        hf_y_keys=(y_gt_key,),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_dataloader = dataset.get_dataloader(
        space_id=test_space_id,
        hf_x_keys=None,
        hf_y_keys=(y_gt_key,),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    model.fit(train_dataloader=train_dataloader, **trainer_params)

    return model.score(dataloader=test_dataloader)


def attach_decoder(
    dataset: DatasetView,
    train_space_id: str,
    test_space_id: str,
    y_gt_key: str,
    model_builder: Callable[[str], LatentisModule],
    train_model: bool,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    exists_ok: bool = False,
) -> Sequence[Dict[str, float]]:
    model = model_builder().to(device)
    model_perfs = None
    if train_model:
        model_perfs = fit_model(
            model=model,
            dataset=dataset,
            train_space_id=train_space_id,
            test_space_id=test_space_id,
            y_gt_key=y_gt_key,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        print(model_perfs)

    train_space = space_index.load_item(item_key=train_space_id)

    try:
        train_space.decoders.add_item(item=model)
    except FileExistsError as e:
        if not exists_ok:
            raise e


if __name__ == "__main__":
    for (dataset_name, label_feature), hf_encoder_name in itertools.product(
        [("trec", "coarse_label"), ("imdb", "label"), ("ag_news", "label")],
        ["bert-base-cased", "bert-base-uncased", "roberta-base"],
    ):
        dataset = DatasetView.load_from_disk(DATA_DIR / dataset_name)

        res = attach_decoder(
            dataset=dataset,
            train_space_id=(
                space_key := space_index.get_item_key(split="train", layer=12, **{"model/hf_name": hf_encoder_name})
            ),
            test_space_id=space_index.get_item_key(split="test", layer=12, **{"model/hf_name": hf_encoder_name}),
            y_gt_key=label_feature,
            model_builder=lambda: Classifier(
                input_dim=space_index.load_item(item_key=space_key).shape[1],  # TODO add this a space property
                num_classes=len(dataset.hf_dataset["train"].features[label_feature].names),
                deep=True,
                bias=True,
                x_feature="encodings_key",
                y_feature="hf_y_keys",
                first_activation=nn.Tanh(),
                second_activation=nn.ReLU(),
                first_projection_dim=None,
            ),
            train_model=True,
            batch_size=128,
            num_workers=0,
            device=torch.device("cpu"),
            exists_ok=True,  # TODO: careful here
        )

        # space = space_index.load_item(split="train", layer=12, **{"model/hf_name": "bert-base-uncased"})
        # decoder = space.decoders.load_item()

        # dataloader = dataset.get_dataloader(
        #     space_id=space_index.get_item_key(
        #         split="test", layer=12, **{"model/hf_name": "bert-base-uncased"}
        #     ),
        #     hf_x_keys=None,
        #     hf_y_keys=(label_feature,),
        #     batch_size=128,
        #     shuffle=True,
        #     num_workers=0,
        # )

        # print(decoder.score(dataloader))
