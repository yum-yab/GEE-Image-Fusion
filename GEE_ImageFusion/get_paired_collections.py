# -*- coding: utf-8 -*-
"""
Author: Ty Nietupski (ty.nietupski@oregonstate.edu)

Functions for image preprocessing and data organization:
    - mask landsat
    - mask modis
    - addNDVI
    - etmToOli
    - get paired image collections (getPaired)
    - reorganize paired collection to units for prediction (makeSubCollections)

This script contains the functions used to acquire, preprocess, and organize
all of the Landsat and MODIS images over a given period of time. These
functions should be run first, after defining some global variables. Functions
in prep_functions.py and core_functions.py follow these. An example of
the use of these functions can be found in Predict_L8.py.


The MIT License

Copyright © 2021 Ty Nietupski

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import ee

from typing import Dict, List, Optional

import pandas as pd

##############################################################################
# MASKING, INDEX CALCULATION, L5 & L7 TO L8 HARMONIZATION
##############################################################################

# converstion from (E)TM (used in LANDSAT5 and 7) to OLI (used in LANDSAT8)
# taken from Roy et al. 2016 https://doi.org/10.1016/j.rse.2015.12.024
# format: {"band_name": [offset, factor]}
# swir1 is for ~1.61 μm, swir2 for ~2.21 μm
LANDSAT_ETM_CORRECTION = {
    "blue": [0.0003, 0.8474],
    "green": [0.0088, 0.8483],
    "red": [0.0061, 0.9047],
    "nir": [0.0412, 0.8462],
    "swir1": [0.0254, 0.8937],
    "swir2": [0.0172, 0.9071],
}


# from https://github.com/google/earthengine-catalog/blob/64a5942e296ee5f803972564fef6abaa14986898/pipelines/landsat.py#L64


def prep_c2sr_l4l5l7(image: ee.Image) -> ee.Image:
    """Scale and mask L5-L7 C2 SR."""
    # Scale optical bands
    optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)

    # Select cloud-free land and water pixels.
    qa = image.select("QA_PIXEL")
    # Clear if bits 0-5 are zero.
    mask1 = qa.bitwiseAnd(int("111111", 2)).eq(0)
    # Good snow/shadow/cloud confidence if bits 8-13 are equal to 010101.
    mask2 = qa.rightShift(8).bitwiseAnd(int("111111", 2)).eq(int("010101", 2))

    # Remove pixels marked as saturated or out of range.
    mask3 = image.select("QA_RADSAT").eq(0)
    mask4 = optical_bands.reduce(ee.Reducer.min()).gt(0)
    mask5 = optical_bands.reduce(ee.Reducer.max()).lt(1)
    # Mark hazy pixels using an empirical AOD threshold.
    mask6 = image.select("SR_ATMOS_OPACITY").unmask(-1).lt(300)

    # Combine masks
    final_mask = mask1.And(mask2).And(mask3).And(mask4).And(mask5).And(mask6)

    # Add scaled bands, apply mask, and ensure float type
    return (
        image.addBands(optical_bands, None, True)
        .updateMask(final_mask)
        .toFloat()
        .copyProperties(image, ["system:time_start"])
    )


def prepare_c2sr_l8l9(image: ee.Image) -> ee.Image:
    """Scale and mask L8-L9 C2 SR."""
    optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    #   thermal_band = image.select('ST_B10').multiply(0.00341802).add(149.0)

    # Insert the scaled bands back into the original image container.
    #   scaled = optical_bands.addBands(thermal_band).select(
    #       ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10'],
    #       ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal'],
    #   )

    # Select cloud free land and water pixels.
    qa = image.select(["QA_PIXEL"])
    # Clear if bits 0-5 are zero.
    mask1 = qa.bitwiseAnd(int("111111", 2)).eq(0)
    # Good snow/shadow/cloud/cirrus confidence if bit pairs 8-15 are each 01.
    mask2 = qa.rightShift(8).bitwiseAnd(int("11111111", 2)).eq(int("01010101", 2))

    # Remove pixels marked as saturated or out of range.
    mask3 = image.select("QA_RADSAT").eq(0)
    mask4 = optical_bands.reduce(ee.Reducer.min()).gt(0)
    mask5 = optical_bands.reduce(ee.Reducer.max()).lt(1)

    # Remove high aerosol pixels (bits 6-7 == 11).
    mask6 = image.select(["SR_QA_AEROSOL"]).rightShift(6).neq(int("11", 2))

    # Put the new bands back into the original image container and mask them.
    return (
        image.addBands(optical_bands, None, True)
        .updateMask(mask1.And(mask2).And(mask3).And(mask4).And(mask5).And(mask6))
        .copyProperties(image, ["system:time_start"])
    )


def scaleMODIS(image: ee.Image) -> ee.Image:
    """
    Scales MODIS bands by scale factor and ensures float type.
    """
    scaled = image.multiply(0.0001)
    return image.addBands(scaled, None, True).toFloat()


def maskLandsat(image):
    """
    Mask cloud, shadow, and snow with fmask and append the percent of pixels \
    masked as new image property.

    Parameters
    ----------
    image : image.Image
        Landsat image with qa band.

    Returns
    -------
    image.image
        Masked landsat image with CloudSnowMaskedPercent property.

    """
    # Bits 3 and 5 are cloud shadow and cloud, respectively. 4 is snow
    cloudShadowBitMask = 1 << 3
    cloudsBitMask = 1 << 5
    snowBitMask = 1 << 4

    # Get the pixel QA band.
    qa = image.select("pixel_qa")

    # make mask
    mask = (
        qa.bitwiseAnd(cloudShadowBitMask)
        .eq(0)
        .And(qa.bitwiseAnd(cloudsBitMask).eq(0))
        .And(qa.bitwiseAnd(snowBitMask).eq(0))
    )

    # mask the mask with the mask...
    maskedMask = mask.updateMask(mask)

    # count the number of nonMasked pixels
    maskedCount = maskedMask.select(["pixel_qa"]).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=image.geometry(),
        scale=ee.Number(30),
        maxPixels=ee.Number(4e10),
    )

    # count the total number of pixels
    origCount = image.select(["blue"]).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=image.geometry(),
        scale=ee.Number(30),
        maxPixels=ee.Number(4e10),
    )

    # calculate the percent of masked pixels
    percent = (
        ee.Number(origCount.get("blue"))
        .subtract(maskedCount.get("pixel_qa"))
        .divide(origCount.get("blue"))
        .multiply(100)
        .round()
    )

    # Return the masked image with new property and time stamp
    return (
        image.updateMask(mask)
        .set("CloudSnowMaskedPercent", percent)
        .copyProperties(image, ["system:time_start"])
    )


def maskMODIS(image):
    """
    Mask snow covered and extremely high albedo areas from the modis images.

    Parameters
    ----------
    image : image.Image
        MODIS image.

    Returns
    -------
    image.image
        Masked MODIS image.

    """
    # calculate snow water index for the image
    swi = image.expression(
        "(green * (nir - swir1)) / ((green + nir) * (nir + swir1))",
        {
            "green": image.select(["green"]),
            "nir": image.select(["nir"]),
            "swir1": image.select(["swir1"]),
        },
    ).rename("swi")

    # mask out values of swi above 0.1
    mask = swi.lt(0.1)

    return image.updateMask(mask).copyProperties(
        image, ["system:time_start", "system:id"]
    )


def addNDVI(image):
    """
    Mask snow covered and extremely high albedo areas from the modis images.

    Parameters
    ----------
    image : image.Image
        Landsat or MODIS image with bands named 'nir' and 'red'.

    Returns
    -------
    image.image
        Image with additional NDVI band.
    """
    # calculate NDVI
    ndvi = image.normalizedDifference(["nir", "red"]).select(["nd"], ["ndvi"])

    return image.addBands(ndvi)


def etmToOli(img):
    """
    Calibrate the NDVI values so that they are more similar to OLI NDVI.

    Parameters
    ----------
    img : image.Image
        Landsat 5 or 7 image.

    Returns
    -------
    image.image
        Adjusted Landsat image.

    """
    # coefficients from Roy et al. 2016
    coefficients = {
        "beta_0": ee.Image.constant([0.0235]),
        "beta_1": ee.Image.constant([0.9723]),
    }

    return (
        img.multiply(coefficients["beta_1"])
        .add(coefficients["beta_0"])
        .toFloat()
        .copyProperties(img, ["system:time_start", "system:id", "DOY"])
    )


def get_combined_landsat(
    wrs_path: int,
    wrs_row: int,
    start_date: str,
    end_date: str,
    landsat_band_mapping: Dict,
    common_bands: List,
    region: Optional[ee.Geometry] = None,
    cloud_cover_limit: int = 20,
    include_l7_slc: bool = False,
) -> ee.ImageCollection:
    """Retrieves and combines Landsat 5, 7, and 8/9 Surface Reflectance imagery.

    This function filters Landsat collections by date, WRS path/row, and optionally
    clips them to a specified region. It applies scaling and masking functions
    (prep_c2sr_l4l5l7 for L5/L7 and prepare_c2sr_l8l9 for L8/L9) and harmonizes
    L5/L7 bands to be consistent with L8/L9 using the `l5l7_to_oli` function.
    This code is largely based on the earthengine collection building code.

    Args:
        wrs_path (int): The Worldwide Reference System (WRS) path number.
        wrs_row (int): The Worldwide Reference System (WRS) row number.
        start_date (str): The start date for filtering images (e.g., 'YYYY-MM-DD').
        end_date (str): The end date for filtering images (e.g., 'YYYY-MM-DD').
        landsat_band_mapping (Dict): A dictionary mapping source band indices
            to target band names for Landsat 5/7 and 8/9.
        common_bands (List): A list of common band names to select across all
            Landsat sensors.
        region (Optional[ee.Geometry], optional): An optional Earth Engine Geometry
            to clip the images. Defaults to None.
        cloud_cover_limit (int, optional): The maximum cloud cover percentage
            allowed for images. Defaults to 20.
        include_l7_slc (bool, optional): Whether to include Landsat 7 imagery
            after its SLC failure (2003-05-31). Defaults to False.

    Returns:
        ee.ImageCollection: An Earth Engine ImageCollection containing combined,
            preprocessed, and harmonized Landsat imagery.
    """

    l7_slc_failure_date = "2003-05-31"
    if include_l7_slc:
        l7_end_date = end_date
    else:
        # If end_date is after SLC failure, cap L7 data at the failure date.
        l7_end_date = min(end_date, l7_slc_failure_date)



    landsat57_source_bn = ee.List(list(landsat_band_mapping.keys()))
    landsat_target_bn = ee.List(list(landsat_band_mapping.values()))

    # we need to shift landsat bands due to the addition of ultra blue band in l8
    landsat8_source_bn = ee.List([i + 1 for i in landsat_band_mapping.keys()])

    common_bands = ee.List(common_bands)

    ee_landsat_etm_correction_coeffs = ee.Dictionary(LANDSAT_ETM_CORRECTION)

    def l5l7_to_oli(image):
        band_names = image.bandNames()

        def correct_band(bandname):
            ee_name = ee.String(bandname)

            has_coeffs = ee_landsat_etm_correction_coeffs.contains(ee_name)

            return ee.Algorithms.If(
                has_coeffs,
                image.select(ee_name)
                .add(
                    ee.Image.constant(
                        ee.List(ee_landsat_etm_correction_coeffs.get(ee_name)).get(0)
                    )
                )
                .multiply(
                    ee.Image.constant(
                        ee.List(ee_landsat_etm_correction_coeffs.get(ee_name)).get(1)
                    )
                ),
                image,
            )

        return (
            ee.ImageCollection.fromImages(band_names.map(correct_band))
            .toBands()
            .rename(band_names)
            .copyProperties(image, ["system:time_start", "system:id"])
        )

    def optional_clipping(img):
        if region is None:
            return img
        else:
            return img.clip(region)

    l5 = (
        ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
        .filterDate(start_date, end_date)
        .filter("WRS_ROW < 122")  # Remove night-time images.
        .filterMetadata("WRS_PATH", "equals", wrs_path)
        .filterMetadata("WRS_ROW", "equals", wrs_row)
        .filterMetadata("CLOUD_COVER", "less_than", cloud_cover_limit)
        .map(optional_clipping)
        .map(prep_c2sr_l4l5l7)
        .select(landsat57_source_bn, landsat_target_bn)
        .select(common_bands)
        .map(l5l7_to_oli)
    )

    l8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        # Images before May 1 had some pointing issues.
        .filterDate("2013-05-01", "2099-01-01")
        .filterDate(start_date, end_date)
        .filter(ee.Filter.neq("NADIR_OFFNADIR", "OFFNADIR"))
        .filter("WRS_ROW < 122")  # Remove night-time images.
        .filterMetadata("WRS_PATH", "equals", wrs_path)
        .filterMetadata("WRS_ROW", "equals", wrs_row)
        .filterMetadata("CLOUD_COVER", "less_than", cloud_cover_limit)
        .map(optional_clipping)
        .map(prepare_c2sr_l8l9)
        .select(landsat8_source_bn, landsat_target_bn)
        .select(common_bands)
    )

    sr = l5.merge(l8)

    if min(start_date, l7_end_date) != l7_end_date:
        l7 = (
            ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
            .filterDate("1984-01-01", "2017-01-01")  # Orbital drift after 2017.
            .filterDate(start_date, l7_end_date)
            .filter("WRS_ROW < 122")  # Remove night-time images.
            .filterMetadata("WRS_PATH", "equals", wrs_path)
            .filterMetadata("WRS_ROW", "equals", wrs_row)
            .filterMetadata("CLOUD_COVER", "less_than", cloud_cover_limit)
            .map(optional_clipping)
            .map(prep_c2sr_l4l5l7)
            .select(landsat57_source_bn, landsat_target_bn)
            .select(common_bands)
            .map(l5l7_to_oli)
        )

        sr = sr.merge(l7)

    return sr


##############################################################################
# FILTER AND PAIR IMAGES
##############################################################################


def get_paired_collections(
    wrs_path: int,
    wrs_row: int,
    start_date: str,
    end_date: str,
    landsat_band_mapping: Dict,
    modis_band_mapping: Dict,
    common_bands: List,
    region: Optional[ee.Geometry] = None,
    modisCollection: str = "MODIS/061/MCD43A4",
    cloud_cover_limit: int = 20,
    include_l7: bool = True,
    modis_unpaired_sample_rate: Optional[int] = None,
) -> (ee.ImageCollection, ee.ImageCollection, ee.ImageCollection):
    """
    Create a list of image collections. Landsat and MODIS with low cloud cover\
    from the same date and the MODIS images between these pairs.

    Parameters
    ----------
    wrs_path : int
        The Worldwide Reference System (WRS) path number.
    wrs_row : int
        The Worldwide Reference System (WRS) row number.
    start_date : str
        Start date of fusion timeframe.
    end_date : str
        End date of the fusion timeframe.
    landsat_band_mapping : Dict
        A dictionary mapping source band indices to target band names for Landsat.
    modis_band_mapping : Dict
        A dictionary mapping source band indices to target band names for MODIS.
    common_bands : List
        A list of common band names to select across both Landsat and MODIS sensors.
    region : Optional[ee.Geometry], optional
        An optional Earth Engine Geometry to clip the images. Defaults to None.
    modisCollection : str, optional
        MODIS collection https://developers.google.com/earth-engine/datasets
    cloud_cover_limit : int, optional
        The maximum cloud cover percentage allowed for Landsat images. Defaults to 20.
    include_l7 : bool, optional
        Whether to include Landsat 7 imagery in the combined collection. Defaults to True.
    modis_unpaired_sample_rate : Optional[int], optional
        If provided, samples the unpaired MODIS collection at this rate. Defaults to None.


    Returns
    -------
    python tuple
        Each element in this tuple is an ee.ImageCollection. 
        The first elements is the Landsat Collection occuring on the same date as the second Element (the Modis Collection) 
        the last element is the MODIS images between each of the pair dates.

    """

    def optional_clipping_modis(img):
        if region is None:
            return img
        else:
            return img.clip(region)

    landsat_collection = get_combined_landsat(
        wrs_path=wrs_path,
        wrs_row=wrs_row,
        start_date=start_date,
        end_date=end_date,
        landsat_band_mapping=landsat_band_mapping,
        common_bands=common_bands,
        region=region,
        cloud_cover_limit=cloud_cover_limit,
        include_l7_slc=include_l7,
    ).map(
        # this is necessary to align with modis images, which are set on midnight (landsat are set some time during the day)
        lambda image: image.setMulti(
            {
                "system:time_start": ee.Date(image.date().format("y-M-d")).millis(),
                "DOY": image.date().format("D"),
            }
        )
    )

    # get modis images
    modis = (
        ee.ImageCollection(modisCollection)
        .filterDate(start=start_date, end=end_date)
        .map(optional_clipping_modis)
        .select(
            ee.List(list(modis_band_mapping.keys())),
            ee.List(list(modis_band_mapping.values())),
        )
        .map(scaleMODIS)
        .map(maskMODIS)
        .map(lambda image: image.set("DOY", image.date().format("D")))
        .select(ee.List(common_bands))
    )

    # filter the two collections by the date property
    dayfilter = ee.Filter.equals(
        leftField="system:time_start", rightField="system:time_start"
    )

    # define simple join
    pairedJoin = ee.Join.simple()
    # define inverted join to find modis images without landsat pair
    invertedJoin = ee.Join.inverted()

    # create collections of paired landsat and modis images
    landsatPaired = pairedJoin.apply(landsat_collection, modis, dayfilter)
    modisPaired = pairedJoin.apply(modis, landsat_collection, dayfilter)
    modisUnpaired = invertedJoin.apply(modis, landsat_collection, dayfilter)

    if modis_unpaired_sample_rate is not None:
        unpaired_list = modisUnpaired.toList(modisUnpaired.size())
        indices = ee.List.sequence(
            0, unpaired_list.size().subtract(1), modis_unpaired_sample_rate
        )
        sampled_list = indices.map(lambda i: unpaired_list.get(i))
        modisUnpaired = ee.ImageCollection.fromImages(sampled_list)

    return (landsatPaired, modisPaired, modisUnpaired)


##############################################################################
# CREATE SUBCOLLECTIONS FOR EACH SET OF LANDSAT/MODIS PAIRS
##############################################################################


def getDates(image, empty_list):
    """
    Get date from image and append to list.

    Parameters
    ----------
    image : image.Image
        Any earth engine image.
    empty_list : ee_list.List
        Earth engine list object to append date to.

    Returns
    -------
    updatelist : ee_list.List
        List with date appended to the end.

    """
    # get date and update format
    date = ee.Image(image).date().format("yyyy-MM-dd")

    # add date to 'empty list'
    updatelist = ee.List(empty_list).add(date)

    return updatelist


def makeSubcollections(paired):
    """
    Reorganize the list of collections into a list of lists of lists. Each\
    list within the list will contain 3 lists. The first of these three will\
    have the earliest and latest Landsat images. The second list will have the\
    earliest and latest MODIS images. The third list will have all the MODIS\
    images between the earliest and latest pairs.\
    (e.g. L8 on 05/22/2017 & 06/23/2017, MOD 05/23/2017 & 06/23/2017,\
     MOD 05/23/2017 through 06/22/2017).

    Parameters
    ----------
    paired : python List
        List of image collections. 1. Landsat pairs, 2. MODIS pairs, and\
        3. MODIS between each of the pairs.

    Returns
    -------
    ee_list.List
        List of lists of lists.

    """

    def getSub(ind):
        """
        Local function to create individual subcollection.

        Parameters
        ----------
        ind : int
            Element of the list to grab.

        Returns
        -------
        ee_list.List
            List of pairs lists for prediction 2 pairs and images between.

        """
        # get landsat images
        lan_01 = (
            paired[0]
            .filterDate(
                ee.List(dateList).get(ind),
                ee.Date(ee.List(dateList).get(ee.Number(ind).add(1))).advance(1, "day"),
            )
            .toList(2)
        )
        # get modis paired images
        mod_01 = (
            paired[1]
            .filterDate(
                ee.List(dateList).get(ind),
                ee.Date(ee.List(dateList).get(ee.Number(ind).add(1))).advance(1, "day"),
            )
            .toList(2)
        )
        # get modis images between these two dates
        mod_p = paired[2].filterDate(
            ee.List(dateList).get(ind),
            ee.Date(ee.List(dateList).get(ee.Number(ind).add(1))).advance(1, "day"),
        )

        mod_p = mod_p.toList(mod_p.size())

        # combine collections to one object
        subcollection = ee.List([lan_01, mod_01, mod_p])

        return subcollection

    # empty list to store dates
    empty_list = ee.List([])

    # fill empty list with dates
    dateList = paired[0].iterate(getDates, empty_list)

    # filter out sub collections from paired and unpaired collections
    subcols = ee.List.sequence(0, ee.List(dateList).length().subtract(2)).map(getSub)

    return subcols
