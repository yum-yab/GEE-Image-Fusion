#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Author: Ty Nietupski (ty.nietupski@oregonstate.edu)
"""

from .core_functions import (
    calcConversionCoeff,
    calcSpatDist,
    calcSpecDist,
    calcWeight,
    predictLandsat,
)
from .get_paired_collections import (
    addNDVI,
    etmToOli,
    get_combined_landsat,
    get_paired_collections,
    getDates,
    makeSubcollections,
    maskLandsat,
    maskMODIS,
    prep_c2sr_l4l5l7,
    prepare_c2sr_l8l9,
    scaleMODIS,
)
from .prep_functions import (
    prepLandsat,
    prepMODIS,
    registerImages,
    threshold,
    threshMask,
)
from .scale_fusion import (
    LandsatFusionComputationConfig,
    compute_year_for_tile,
    handle_subcollection,
)

__all__ = [
    # core_functions
    "calcSpecDist",
    "calcSpatDist",
    "calcWeight",
    "calcConversionCoeff",
    "predictLandsat",
    # prep_functions
    "registerImages",
    "threshold",
    "threshMask",
    "prepMODIS",
    "prepLandsat",
    # get_paired_collections
    "prep_c2sr_l4l5l7",
    "prepare_c2sr_l8l9",
    "scaleMODIS",
    "maskLandsat",
    "maskMODIS",
    "addNDVI",
    "etmToOli",
    "get_combined_landsat",
    "get_paired_collections",
    "getDates",
    "makeSubcollections",
    # scale_fusion
    "LandsatFusionComputationConfig",
    "handle_subcollection",
    "compute_year_for_tile",
]