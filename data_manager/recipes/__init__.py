#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
TS-DIA Custom Recipes Package

This package will contain custom dataset recipes and processing utilities.
Currently empty but ready for future expansion.
"""

# This package is currently empty but ready for custom recipes
# Future modules can be added here for dataset-specific processing

from .ava_avd import download_ava_avd, prepare_ava_avd
from .ego4d import download_ego4d, prepare_ego4d
from .libriheavy_mix import download_libriheavy_mix, prepare_libriheavy_mix
from .mswild import download_mswild, prepare_mswild
from .voxconverse import download_voxconverse, prepare_voxconverse

__all__ = [
    "download_mswild",
    "prepare_mswild",
    "download_voxconverse",
    "prepare_voxconverse",
    "download_ava_avd",
    "prepare_ava_avd",
    "download_libriheavy_mix",
    "prepare_libriheavy_mix",
    "download_ego4d",
    "prepare_ego4d",
]

__version__ = "0.1.0"
