import logging
import os
from pathlib import Path
from typing import Any, List, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Callback
from torch import nn

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import test_nn_template  # noqa
from test_nn_template.data.datamodule import MetaData

pylogger = logging.getLogger(__name__)


def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def instantiate(_target_: str, **kwargs) -> Any:
    return hydra.utils.instantiate(_target_, _recursive_=False, **kwargs)


class Runner(object):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.template_core = None
        self.callbacks = None
        self.storage_dir = None
        self.logger = None
        self.trianer = None
        self.build_callbacks()
        self.build_logger()
        self.build_trainer()

    def setup(self):
        seed_index_everything(self.cfg.train)
        self.fast_dev_run: bool = self.cfg.train.trainer.fast_dev_run
        if self.fast_dev_run:
            pylogger.info(
                f"Debug mode <{self.cfg.train.trainer.fast_dev_run=}>. Forcing debugger friendly configuration!"
            )
            # Debuggers don't like GPUs nor multiprocessing
            self.cfg.train.trainer.gpus = 0
            self.cfg.data.datamodule.num_workers.train = 0
            self.cfg.data.datamodule.num_workers.val = 0
            self.cfg.data.datamodule.num_workers.test = 0
        self.cfg.core.tags = enforce_tags(self.cfg.core.get("tags", None))

    @classmethod
    def get_datamodule(cls, cfg: DictConfig) -> pl.LightningDataModule:
        pylogger.info(f"Instantiating <{cfg.data.datamodule['_target_']}>")
        datamodule: pl.LightningDataModule = instantiate(cfg.data.datamodule)

        metadata: Optional[MetaData] = getattr(datamodule, "metadata", None)
        if metadata is None:
            pylogger.warning(f"No 'metadata' attribute found in datamodule <{datamodule.__class__.__name__}>")
        return datamodule

    @classmethod
    def get_arch(cls, cfg: DictConfig, **kwargs) -> nn.Module:
        pylogger.info(f"Instantiating <{cfg.model.arch['_target_']}>")
        arch: nn.Module = instantiate(cfg.model.arch, **kwargs)
        return arch

    @classmethod
    def get_model(cls, cfg: DictConfig, arch: nn.Module, **kwargs) -> pl.LightningModule:
        pylogger.info(f"Instantiating <{cfg.model.module['_target_']}>")
        model: pl.LightningModule = instantiate(cfg.model.module, model=arch, **cfg.optim, **kwargs)
        return model

    def build_callbacks(self) -> None:
        self.template_core: NNTemplateCore = NNTemplateCore(
            restore_cfg=self.cfg.train.get("restore", None),
        )
        self.callbacks: List[Callback] = build_callbacks(self.cfg.train.callbacks, self.template_core)
        self.storage_dir: str = self.cfg.core.storage_dir

    def build_logger(self) -> None:
        self.logger: NNLogger = NNLogger(
            logging_cfg=self.cfg.train.logging, cfg=self.cfg, resume_id=self.template_core.resume_id
        )

    def build_trainer(self):
        pylogger.info("Instantiating the <Trainer>")
        self.trainer = pl.Trainer(
            default_root_dir=self.storage_dir,
            plugins=[NNCheckpointIO(jailing_dir=self.logger.run_dir)],
            logger=self.logger,
            callbacks=self.callbacks,
            **self.cfg.train.trainer,
        )

    def train(self, model: pl.LightningModule, datamodule: pl.LightningDataModule) -> str:
        pylogger.info("Starting training!")
        self.trainer.fit(model=model, datamodule=datamodule, ckpt_path=self.template_core.trainer_ckpt_path)

        if self.fast_dev_run:
            pylogger.info("Skipping testing in 'fast_dev_run' mode!")
        else:
            if "test" in self.cfg.data.datasets and self.trainer.checkpoint_callback.best_model_path is not None:
                pylogger.info("Starting testing!")
                self.trainer.test(datamodule=datamodule)

        # Logger closing to release resources/avoid multi-run conflicts
        if self.logger is not None:
            self.logger.experiment.finish()

        return self.logger.run_dir

    def run(self) -> str:
        datamodule = self.get_datamodule(cfg=self.cfg)
        num_classes = len(datamodule.metadata.class_vocab)
        arch = self.get_arch(cfg=self.cfg, num_classes=num_classes)
        model = self.get_model(cfg=self.cfg, arch=arch, metadata=datamodule.metadata)
        return self.train(datamodule=datamodule, model=model)


class WandbHandler(object):
    @classmethod
    def get_run(cls, run_id: str):
        api = wandb.Api()
        return api.run(run_id)

    @classmethod
    def download_run_checkpoint(cls, run_id: str, replace: bool = True) -> None:
        run = cls.get_run(run_id)
        checkpoints = [file.name for file in run.files() if "ckpt.zip" in file.name]
        assert len(checkpoints) == 1, f"{len(checkpoints)} checkpoints found for run '{run_id}', expected 1"
        checkpoint = checkpoints[0]
        print(f"Downloading {checkpoint}")
        run.file(checkpoint).download(root=run.name, replace=replace)
        return os.path.join(run.name, checkpoint)

    @classmethod
    def get_run_checkpoint(cls, run_id: str):
        checkpoint = cls.download_run_checkpoint(run_id)
        print(f"Loading {checkpoint}")
        checkpoint = NNCheckpointIO.load(path=Path(checkpoint))
        checkpoint["cfg"] = DictConfig(checkpoint["cfg"])
        return checkpoint

    @classmethod
    def load_run_model_checkpoint(cls, run_id: str):
        checkpoint = cls.get_run_checkpoint(run_id)
        cfg = checkpoint["cfg"]
        metadata = checkpoint["metadata"]
        arch: nn.Module = instantiate(cfg.model.arch, num_classes=len(metadata.class_vocab))
        model: pl.LightningModule = instantiate(cfg.model.module, metadata=metadata, model=arch, **cfg.optim)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    @classmethod
    def load_run_datamodule(cls, run_id: str, cfg_func: callable = lambda x: x):
        checkpoint = cls.get_run_checkpoint(run_id)
        cfg = cfg_func(checkpoint["cfg"])
        datamodule: pl.LightningDataModule = instantiate(cfg.data.datamodule)
        _: Optional[MetaData] = getattr(datamodule, "metadata", None)
        return datamodule

    @classmethod
    def get_runs(cls, entity: str = "fernandoezequiel512", project: str = "test_nn_template"):
        api = wandb.Api()
        return api.runs(f"{entity}/{project}")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    runner = Runner(cfg=cfg)
    return runner.run()


if __name__ == "__main__":
    main()
