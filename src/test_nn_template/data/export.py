import json
import os
from typing import Callable, Optional

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from wandb.apis.public import Run


class EmbeddingsSaver(object):
    def __init__(
        self,
        run: Run,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        source: str,
        class_to_index: dict,
        transform: Optional[Callable] = None,
    ):
        self.transform = transform
        self.train = train_dataloader
        self.test = test_dataloader
        self.run = run
        self.run_id = run.name
        self.metadata = self.build_metadata(source, class_to_index)

    def build_metadata(self, source, class_to_index):
        return {
            "wdb_entity": self.run.entity,
            "wdb_project": self.run.project,
            "wdb_run_id": self.run.name,
            "source": source,
            "class_to_idx": class_to_index,
        }

    @classmethod
    def save_dataloader(cls, dataloader: DataLoader, transform: Callable, embeddings_dir: str, labels_dir: str) -> None:
        for ix, (x, y) in tqdm(enumerate(dataloader), desc="Saving embedding", total=len(dataloader)):
            for jx, emb in enumerate(transform(x)):
                n = ix * dataloader.batch_size + jx
                emb_path = os.path.join(embeddings_dir, f"{n}.pt")
                lab_path = os.path.join(labels_dir, f"{n}.pt")
                cls.save_tensor(emb, emb_path)
                cls.save_tensor(y[jx], lab_path)

    @classmethod
    def save_tensor(cls, tensor: torch.Tensor, path: str) -> None:
        torch.save(tensor.detach(), path)

    def save_dict(self, _dict: dict, path: str) -> None:
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(_dict, f)

    def save(self, root: str) -> None:
        for slice, loader in [("train", self.train), ("test", self.test)]:
            curr_root = os.path.join(root, self.run_id, slice)
            metadata_dir, embeddings_dir, labels_dir = self.build_folders_struct(curr_root)
            self.save_dict(self.metadata, metadata_dir)
            self.save_dataloader(loader, self.transform, embeddings_dir, labels_dir)

    @classmethod
    def build_folders_struct(cls, root: str) -> None:
        metadata_dir = os.path.join(root, "metadata")
        embeddings_dir = os.path.join(root, "data", "embeddings")
        labels_dir = os.path.join(root, "data", "labels")
        os.makedirs(metadata_dir)
        os.makedirs(embeddings_dir)
        os.makedirs(labels_dir)
        return metadata_dir, embeddings_dir, labels_dir
