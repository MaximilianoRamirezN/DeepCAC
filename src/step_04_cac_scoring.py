from typing import Tuple

import SimpleITK as sitk

SITKImage = sitk.SimpleITK.Image

from util import (
    read_image,
    get_ag,
    get_ag_class,
)


def cac_scoring(
    img_sitk: SITKImage,
    cacseg_sitk: SITKImage,
    threshold: float = 0.1,
    min_n_pixels: int = 3,
    ag_div: int = 3,
) -> Tuple[float, int]:
    """
    Perform the fourth step of the DeepCAC pipeline - CAC scoring.

    Parameters
    ----------
    img_sitk: SITKImage
        the CT image to be processed
    cacseg_sitk: SITKImage
        the CAC segmentation mask of the third step of the DeepCAC pipeline
    threshold: float, optional
        threshold applied to the CAC segmentation mask
    min_n_pixels: int, optional
        regions smaller than this are ignored for scornig
    ag_div: int, optional
        divider of the pixel volume

    Returns
    -------
    Tuple[float, int]
        the Agatston-Score and the according risk class
    """
    img = sitk.GetArrayFromImage(img_sitk)
    prd = sitk.GetArrayFromImage(cacseg_sitk)

    prd[prd <= threshold] = 0
    prd[prd > 0] = 1

    ag = get_ag(
        img, prd, img_sitk.GetSpacing(), min_n_pixels=min_n_pixels, ag_div=ag_div
    )
    ag_class = get_ag_class(ag)
    return ag, ag_class


if __name__ == "__main__":
    import argparse

    from util import read_image

    parser = argparse.ArgumentParser(
        description="Perform the fourth step - CAC scoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=str, help="the CT image to process")
    parser.add_argument(
        "--cacseg", type=str, help="nrrd image of the cac segmentation step"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="threshold applied to the cac segmentation",
    )
    parser.add_argument(
        "--min_n_pixels", type=int, default=3, help="divider for the pixel volume"
    )
    parser.add_argument(
        "--ag_div",
        type=int,
        default=3,
        help="regions with fewer pixels will be ignored for scoring",
    )
    args = parser.parse_args()

    img_sitk = read_image(args.image)
    cacseg_sitk = read_image(args.cacseg)

    ag, ag_class = cac_scoring(img_sitk, cacseg_sitk, threshold=args.threshold)
    print(f"Agatson score of {ag:.3f} which is class {ag_class}")
