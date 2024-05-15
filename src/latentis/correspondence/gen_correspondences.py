import torch

from latentis.correspondence._base import Correspondence
from latentis.nexus import correspondences_index

if __name__ == "__main__":
    for x_dataset in ["trec", "imdb"]:
        for y_dataset in ["trec", "imdb", "amazon"]:
            for quality in torch.arange(0, 1, 0.1):
                for split in ["train", "test"]:
                    corr = Correspondence(
                        x_dataset=x_dataset,
                        y_dataset=y_dataset,
                        quality=quality.item(),
                        split=split,
                    )
                    correspondences_index.add_item(corr)
