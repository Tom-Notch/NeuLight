#! /usr/bin/env python
#
# Created on Wed Apr 23 2025 00:21:07
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import argparse
import os.path as osp

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from neulight.dataset.free_viewpoint import FreeViewpointDataModule
from neulight.model.neus import NeuSLightningModel
from neulight.utils.files import parse_path
from neulight.utils.files import print_dict
from neulight.utils.files import read_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to the configuration file.",
        default="config/neus.yaml",
    )
    parser.add_argument(
        "--eval",
        "-e",
        action="store_true",
        help="If specified, perform one epoch of validation only.",
    )
    args = parser.parse_args()
    current_dir = osp.dirname(osp.realpath(__file__))
    base_dir = osp.join(current_dir, "..")

    config_path = parse_path(args.config, base_dir)

    print(f"Running NeuS with config file: {config_path}")

    config = read_file(config_path)
    print("config:")
    print_dict(config)

    data_module = FreeViewpointDataModule(**config["dataloader"])

    model = NeuSLightningModel(config["model"])

    config["trainer"]["logger"] = WandbLogger(**config["wandb_logger"])

    trainer = pl.Trainer(**config["trainer"])

    if args.eval:
        trainer.validate(model, data_module)
    else:
        trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
