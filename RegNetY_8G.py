#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test a trained classification model."""

import argparse
import sys

import numpy as np
import pycls.core.losses as losses
import pycls.core.model_builder as model_builder
import pycls.datasets.loader as loader
import pycls.utils.benchmark as bu
import pycls.utils.checkpoint as cu
import pycls.utils.distributed as du
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.multiprocessing as mpu
import pycls.utils.net as nu
import torch
from pycls.core.config import assert_and_infer_cfg, cfg
from pycls.utils.meters import TestMeter


logger = lu.get_logger(__name__)

def log_model_info(model):
    """Logs model info"""
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(mu.params_count(model)))
    logger.info("Flops: {:,}".format(mu.flops_count(model)))
    logger.info("Acts: {:,}".format(mu.acts_count(model)))

def build_model():

    # Load config options
    cfg.merge_from_file('RegNetY-8.0GF_dds_8gpu.yaml')
    cfg.merge_from_list([])
    assert_and_infer_cfg()
    cfg.freeze()
    # Setup logging
    lu.setup_logging()
    # Show the config
    logger.info("Config:\n{}".format(cfg))

    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    # Build the model (before the loaders to speed up debugging)
    model = model_builder.build_model()
    log_model_info(model)

    # Load model weights
    #cu.load_checkpoint('RegNetY-8.0GF_dds_8gpu.pyth', model)
    logger.info("Loaded model weights from: {}".format('RegNetY-8.0GF_dds_8gpu.pyth'))

    del model.head

    return model

