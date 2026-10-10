from omegaconf import OmegaConf
from accel_hydra.utils.config import register_omegaconf_resolvers as register_base_resolvers
from pathlib import Path
from utils.tokenize import get_vocab_size
import sys
from typing import Union
import os

import hydra
import omegaconf

def add(*x):
    return sum(x)


def register_omegaconf_resolvers() -> None:
    """
    Register custom resolvers.
    This function first calls the base resolvers from accel_hydra, then registers additional resolvers specific to UniFlow-Audio.
    """
    register_base_resolvers(clear_resolvers=True)

    OmegaConf.register_new_resolver(
        "get_vocab_size", get_vocab_size, replace=True
    )
    OmegaConf.register_new_resolver("add", add, replace=True)

def generate_config_from_command_line_overrides(
    config_file: Union[str, Path]
) -> omegaconf.DictConfig:
    register_omegaconf_resolvers()

    config_file = Path(config_file).resolve()
    config_name = config_file.name.__str__()
    config_path = config_file.parent.__str__()
    config_path = os.path.relpath(config_path, Path(__file__).resolve().parent)

    overrides = sys.argv[1:]
    with hydra.initialize(version_base=None, config_path=config_path):
        config = hydra.compose(config_name=config_name, overrides=overrides)
    omegaconf.OmegaConf.resolve(config)

    return config