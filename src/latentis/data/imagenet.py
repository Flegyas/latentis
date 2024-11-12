from pathlib import Path

import pandas as pd
from datasets import ClassLabel, Dataset

from latentis import PROJECT_ROOT


def read_imagenet_labels(
    root_dir: Path = None, file_name: str = "ImageNet_mapping.tsv"
):
    file = (root_dir or PROJECT_ROOT / "data") / file_name
    data = pd.read_csv(file, sep="\t")
    assert len(data) == 1000

    data["pos"] = data["synset_id"].str[0]
    data["offset"] = data["synset_id"].str[1:].str.rjust(8, "0")
    data["class_id"] = range(len(data))
    data["openai_lemma"] = data["openai_lemma"].str.strip()

    return data


OPENAI_IMAGENET_TEMPLATES = (
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a photo of many {c}.",
    lambda c: f"a sculpture of a {c}.",
    lambda c: f"a photo of the hard to see {c}.",
    lambda c: f"a low resolution photo of the {c}.",
    lambda c: f"a rendering of a {c}.",
    lambda c: f"graffiti of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a cropped photo of the {c}.",
    lambda c: f"a tattoo of a {c}.",
    lambda c: f"the embroidered {c}.",
    lambda c: f"a photo of a hard to see {c}.",
    lambda c: f"a bright photo of a {c}.",
    lambda c: f"a photo of a clean {c}.",
    lambda c: f"a photo of a dirty {c}.",
    lambda c: f"a dark photo of the {c}.",
    lambda c: f"a drawing of a {c}.",
    lambda c: f"a photo of my {c}.",
    lambda c: f"the plastic {c}.",
    lambda c: f"a photo of the cool {c}.",
    lambda c: f"a close-up photo of a {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a painting of the {c}.",
    lambda c: f"a painting of a {c}.",
    lambda c: f"a pixelated photo of the {c}.",
    lambda c: f"a sculpture of the {c}.",
    lambda c: f"a bright photo of the {c}.",
    lambda c: f"a cropped photo of a {c}.",
    lambda c: f"a plastic {c}.",
    lambda c: f"a photo of the dirty {c}.",
    lambda c: f"a jpeg corrupted photo of a {c}.",
    lambda c: f"a blurry photo of the {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a rendering of the {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"a photo of one {c}.",
    lambda c: f"a doodle of a {c}.",
    lambda c: f"a close-up photo of the {c}.",
    lambda c: f"a photo of a {c}.",
    lambda c: f"the origami {c}.",
    lambda c: f"the {c} in a video game.",
    lambda c: f"a sketch of a {c}.",
    lambda c: f"a doodle of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a low resolution photo of a {c}.",
    lambda c: f"the toy {c}.",
    lambda c: f"a rendition of the {c}.",
    lambda c: f"a photo of the clean {c}.",
    lambda c: f"a photo of a large {c}.",
    lambda c: f"a rendition of a {c}.",
    lambda c: f"a photo of a nice {c}.",
    lambda c: f"a photo of a weird {c}.",
    lambda c: f"a blurry photo of a {c}.",
    lambda c: f"a cartoon {c}.",
    lambda c: f"art of a {c}.",
    lambda c: f"a sketch of the {c}.",
    lambda c: f"a embroidered {c}.",
    lambda c: f"a pixelated photo of a {c}.",
    lambda c: f"itap of the {c}.",
    lambda c: f"a jpeg corrupted photo of the {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a plushie {c}.",
    lambda c: f"a photo of the nice {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the weird {c}.",
    lambda c: f"the cartoon {c}.",
    lambda c: f"art of the {c}.",
    lambda c: f"a drawing of the {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"the plushie {c}.",
    lambda c: f"a dark photo of a {c}.",
    lambda c: f"itap of a {c}.",
    lambda c: f"graffiti of the {c}.",
    lambda c: f"a toy {c}.",
    lambda c: f"itap of my {c}.",
    lambda c: f"a photo of a cool {c}.",
    lambda c: f"a photo of a small {c}.",
    lambda c: f"a tattoo of the {c}.",
)


def get_template_dataset(test_size: float = 0.15, seed: int = 42) -> Dataset:
    imagenet_data = read_imagenet_labels()
    data = {
        "synset_id": [],
        "class_id": [],
        "lemma": [],
        "template_id": [],
        "sample_id": [],
        "text": [],
    }

    for imagenet_synset in imagenet_data.itertuples():
        for template_id, template in enumerate(OPENAI_IMAGENET_TEMPLATES):
            sentence = template(imagenet_synset.openai_lemma)
            data["synset_id"].append(imagenet_synset.synset_id)
            data["class_id"].append(imagenet_synset.class_id)
            data["text"].append(sentence)
            data["template_id"].append(template_id)
            data["sample_id"].append(f"{imagenet_synset.synset_id}_{template_id}")
            data["lemma"].append(imagenet_synset.openai_lemma)

    data = Dataset.from_dict(data)

    class_label = ClassLabel(
        num_classes=len(set(data["class_id"])),
        names=list(set(data["class_id"])),
    )
    data = data.cast_column("class_id", class_label)

    template_label = ClassLabel(
        num_classes=len(set(data["template_id"])),
        names=list(set(data["template_id"])),
    )
    data = data.cast_column("template_id", template_label)

    return data.train_test_split(
        test_size=test_size, seed=seed, stratify_by_column="class_id"
    )


# get_template_dataset()
