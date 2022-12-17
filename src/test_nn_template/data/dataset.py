import json
import os
import random
from collections import defaultdict
from glob import glob

import hydra
import omegaconf
import torch
from torch.utils.data import Dataset, get_worker_info
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split


class MyDataset(Dataset):
    def __init__(self, split: Split, **kwargs):
        super().__init__()
        self.split: Split = split

        # example
        self.mnist = FashionMNIST(
            kwargs["path"],
            train=split == "train",
            download=True,
            transform=kwargs.get("transform"),
        )

    @property
    def class_vocab(self):
        return self.mnist.class_to_idx

    def __len__(self) -> int:
        # example
        return len(self.mnist)

    def __getitem__(self, index: int):
        # example
        return self.mnist[index]

    def __repr__(self) -> str:
        return f"MyDataset({self.split=}, n_instances={len(self)})"


class MyContrastativeDataset(Dataset):
    def __init__(self, split: Split, size: int, **kwargs):
        super().__init__()
        self.split: Split = split
        self.seed = 42
        self.size = size
        # example
        self.mnist = FashionMNIST(
            kwargs["path"],
            train=split == "train",
            download=True,
            transform=kwargs.get("transform"),
        )
        self.build_dataset()

    def choose_examples(self):
        # pool big enough to get close to balancd pos/neg ratio
        n_pool = self.size * 3
        ex_pool = random.choices(range(len(self.mnist)), k=n_pool)  # nosec
        totals_per_class = defaultdict(int)
        for ix in ex_pool:
            _, cls = self.mnist[ix]
            totals_per_class[cls] += 1
        per_class_sampling_weight = {}
        for cls, total in totals_per_class.items():
            per_class_sampling_weight[cls] = (n_pool - total) / total
        per_class_pool_weights = {}
        for cls, weight in tqdm(per_class_sampling_weight.items(), desc="Caching sampling weights"):
            per_class_pool_weights[cls] = [weight if self.mnist[ix][1] == cls else 1 for ix in ex_pool]
        for ix in ex_pool:
            ex1, lab1 = self.mnist[ix]
            jx = random.choices(ex_pool, weights=per_class_pool_weights[lab1], k=1)[0]  # nosec
            ex2, lab2 = self.mnist[jx]
            yield (ex1, ex2, int(lab1 == lab2))

    def build_dataset(self):
        worker_info = get_worker_info()
        if worker_info is not None:  # inside a worker process
            # split workload
            self.seed = get_worker_info.seed + worker_info.id
        random.seed(self.seed)
        examples_generator = self.choose_examples()
        self.dataset = [next(examples_generator) for _ in tqdm(range(self.size), desc="Building contrastative dataset")]

    @property
    def class_vocab(self):
        return {"same class": 1, "different class": 0}

    def __getitem__(self, index: int):
        # example
        return self.dataset[index]

    def __len__(self):
        return self.size

    def __repr__(self) -> str:
        return f"MyContrastativeDataset({self.split=}, {self.size=}, seed={self.seed})"


class MyEmbeddingsDataset(Dataset):
    def __init__(self, split: Split, **kwargs):
        super().__init__()
        self.split: Split = split
        self.root = os.path.join(kwargs["path"], kwargs["run"])
        self.embeddings = glob(os.path.join(self.root, split, "data/embeddings/*.pt"))
        self.labels = glob(os.path.join(self.root, split, "data/labels/*.pt"))
        with open(os.path.join(self.root, split, "metadata", "metadata.json")) as f:
            self.metadata = json.load(f)

    @property
    def class_vocab(self):
        return self.metadata["class_to_idx"]

    def __len__(self) -> int:
        # example
        return len(self.embeddings)

    def __getitem__(self, index: int):
        # example
        emb = torch.load(self.embeddings[index])
        label = torch.load(self.labels[index])
        # label = torch.nn.functional.one_hot(label, num_classes=12)
        return (emb, int(label.detach()))
        # return (torch.rand(2), random.choice(range(9)))

    def __repr__(self) -> str:
        return f"MyDataset({self.split=}, n_instances={len(self)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    cfg.data.datasets.train["size"] = 100
    data: Dataset = hydra.utils.instantiate(
        cfg.data.datasets.train, split="train", path=PROJECT_ROOT / "data", _recursive_=False
    )
    print(data)

    cfg.data.datasets.train["_target_"]: str = cfg.data.datasets.train["_target_"].replace(
        "MyContrastativeDataset", "MyEmbeddingsDataset"
    )

    data: Dataset = hydra.utils.instantiate(
        cfg.data.datasets.train, split="train", path=PROJECT_ROOT / "data", _recursive_=False, run="sage-field-17"
    )
    print(data)

    ex = data[0]
    print(ex)


if __name__ == "__main__":
    main()
