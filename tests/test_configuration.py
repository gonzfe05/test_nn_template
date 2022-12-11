import hydra
from omegaconf import DictConfig

from test_nn_template.run import build_callbacks


def test_configuration_parsing(cfg: DictConfig) -> None:
    assert cfg is not None


def test_callbacks_instantiation(cfg: DictConfig) -> None:
    build_callbacks(cfg.train.callbacks)


def test_model_instantiation(cfg: DictConfig) -> None:
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
    arch = hydra.utils.instantiate(cfg.model.arch, _recursive_=False, num_classes=len(datamodule.metadata.class_vocab))
    hydra.utils.instantiate(cfg.model.module, metadata=datamodule.metadata, _recursive_=False, model=arch, **cfg.optim)


def test_cfg_parametrization(cfg_all: DictConfig):
    assert cfg_all
