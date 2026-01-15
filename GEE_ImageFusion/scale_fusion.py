import ee

from typing import Optional, List

from .prep_functions import registerImages, prepLandsat, prepMODIS
from .core_functions import (
    calcSpecDist,
    calcSpatDist,
    calcWeight,
    calcConversionCoeff,
    predictLandsat,
)

from .get_paired_collections import get_paired_collections, makeSubcollections


class LandsatFusionComputationConfig:
    def __init__(
        self,
        common_bands: List[str],
        kernel_radius: int,
        cover_classes: int = 7,
        register_images: bool = True,
        modis_interp_sample_rate: Optional[int] = None,
        region_clip: Optional[ee.Geometry] = None,
    ):
        self.landsat_band_mapping = dict(
            zip([0, 1, 2, 3, 4, 5], ["blue", "green", "red", "nir", "swir1", "swir2"])
        )
        self.modis_band_mapping = dict(
            zip([2, 3, 0, 1, 5, 6], ["blue", "green", "red", "nir", "swir1", "swir2"])
        )

        self.cover_classes = cover_classes

        self.register_images = register_images

        self.common_bands = common_bands
        self.kernel_radius = kernel_radius
        self.modis_interp_sample_rate = modis_interp_sample_rate
        self.region_clip = region_clip
        # Note: Generally, larger windows are better but as the window size increases,
        # so does the memory requirement and we quickly will surpass the memory
        # capacity of a single node (in testing 13 was max size for single band, and
        # 10 was max size for up to 6 bands)
        self.kernel_radius_ee = ee.Number(kernel_radius)
        self.kernel = ee.Kernel.square(
            radius=self.kernel_radius_ee, units="pixels", normalize=True, magnitude=1
        )
        self.numPixels = ee.Number((2 * kernel_radius + 1) ** 2)


def handle_subcollection(
    subcollection: ee.List,
    fusion_comp_cfg: LandsatFusionComputationConfig,
) -> ee.List:
    # radius of moving window

    landsat_borderpair = ee.List(ee.List(subcollection).get(0))
    modis_borderpair = ee.List(ee.List(subcollection).get(1))
    modis_interp_images = ee.List(ee.List(subcollection).get(2))

    # at this point the collections should be reduced to the  common band names

    common_bands = landsat_borderpair.get(0).bandNames()

    if fusion_comp_cfg.sample_rate is not None:
        indices = ee.List.sequence(
            0,
            modis_interp_images.size().subtract(1),
            fusion_comp_cfg.modis_interp_sample_rate,
        )
        modis_interp_images = indices.map(lambda i: modis_interp_images.get(i))

    if fusion_comp_cfg.register_images:
        landsat_borderpair, modis_borderpair, modis_interp_images = registerImages(
            landsat_borderpair, modis_borderpair, modis_interp_images
        )

    doys = landsat_borderpair.map(
        lambda img: ee.String(ee.Image(img).get("DOY")).cat("_")
    )

    maskedLandsat, pixPositions, pixBN = prepLandsat(
        landsat_borderpair,
        fusion_comp_cfg.kernel,
        fusion_comp_cfg.numPixels,
        common_bands,
        doys,
        fusion_comp_cfg.cover_classes,
    )

    modSorted_t01, modSorted_tp = prepMODIS(
        modis_borderpair,
        modis_interp_images,
        fusion_comp_cfg.kernel,
        fusion_comp_cfg.numPixels,
        common_bands,
        pixBN,
    )

    specDist = calcSpecDist(
        maskedLandsat, modSorted_t01, fusion_comp_cfg.numPixels, pixPositions
    )

    spatDist = calcSpatDist(pixPositions)

    weights = calcWeight(spatDist, specDist)

    coeffs = calcConversionCoeff(
        maskedLandsat, modSorted_t01, doys, fusion_comp_cfg.numPixels, common_bands
    )

    prediction = modSorted_tp.map(
        lambda image: predictLandsat(
            landsat_borderpair,
            modSorted_t01,
            doys,
            ee.List(image),
            weights,
            coeffs,
            common_bands,
            fusion_comp_cfg.numPixels,
        )
    )

    # we always add the first border landsat image and the chosen predictions as collection
    # this ensures no duplications, since the other border will be added by the the following subcollection

    return prediction.insert(0, ee.Image(landsat_borderpair.get(0)))




def compute_year_for_tile(
    wrs_path: int,
    wrs_row: int,
    start_date: str,
    end_date: str,
    year: int,
    fusion_comp_cfg: LandsatFusionComputationConfig,
    cloud_cover_limit: int = 20,
    include_l7_slc: bool = False,
) -> ee.ImageCollection:
    paired = get_paired_collections(
        wrs_path=wrs_path,
        wrs_row=wrs_row,
        start_date=start_date,
        end_date=end_date,
        include_l7=include_l7_slc,
        region=fusion_comp_cfg.region_clip,
        cloud_cover_limit=cloud_cover_limit,
        landsat_band_mapping=fusion_comp_cfg.landsat_band_mapping,
        modis_band_mapping=fusion_comp_cfg.modis_band_mapping,
        common_bands=fusion_comp_cfg.common_bands,
    )

    subcollections = makeSubcollections(paired)

    image_lists_list = subcollections.map(
        lambda subcollection: handle_subcollection(
            subcollection=subcollection,
            fusion_comp_cfg=fusion_comp_cfg,
        )
    )


    return ee.ImageCollection.fromImages(image_lists_list.flatten()).filterDate("{year}-01-01", "{year}-12-31")