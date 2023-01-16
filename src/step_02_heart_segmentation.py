from typing import List, Optional

import numpy as np
import SimpleITK as sitk

SITKImage = sitk.SimpleITK.Image

import step2_heartseg.heartseg_model as heartseg_model

from util import (
    normalize_ct,
    change_spacing,
    get_bounding_box,
    keep_only_the_largest_volume,
    center_crop,
    center_pad,
)


def heart_segmentation(
    img_sitk: SITKImage,
    heartloc_sitk: SITKImage,
    input_spacing: List[int] = [2.04, 2.04, 2.5],
    input_size: List[int] = [112, 128, 128],
    model_weights: str = "../data/step2_heartseg/model_weights/step2_heartseg_model_weights.hdf5",
) -> SITKImage:
    """
    Perform the second step of the DeepCAC pipeline - heart segmentation.

    Parameters
    ----------
    img_sitk: SITKImage
        the CT image to be processed
    heartloc_sitk: SITKImage
        the rough heart segmentation mask of the first step of the DeepCAC pipeline
    input_spacing: List[int], optional
        the CT image is resampled to this spacing before processing with the
        neural network
    input_size: List[int], optional
        after resampling, the image is cropped and padded to this size before
        processing by the neural network
    model_weights: str, optional
        filename of the model weights for the neural network

    Returns
    -------
    SITKImage
        a binary segmentation mask of the heart with the same
        spacing and size as the input image
    """
    # store original spacing and size to recover it for the resulting segmentation
    original_spacing = img_sitk.GetSpacing()
    original_size = img_sitk.GetSize()

    # change spacing for model input
    img_sitk = change_spacing(img_sitk, input_spacing)
    img = sitk.GetArrayFromImage(img_sitk)
    original_shape = img.shape

    # change spacing of segmentation accordingly
    heartloc_sitk = change_spacing(heartloc_sitk, input_spacing)
    img_heartloc = sitk.GetArrayFromImage(heartloc_sitk)
    # threshold segmentation
    img_heartloc[img_heartloc <= 0.5] = 0
    img_heartloc[img_heartloc > 0] = 1
    # compute bounding box of segmented region
    bb = get_bounding_box(img_heartloc)

    # select bounding box in image
    img = img[bb[0] : bb[1], bb[2] : bb[3], bb[4] : bb[5]]
    bb_shape = img.shape

    # pad to expected model input size and normalize
    img = center_pad(img, input_size, constant=-1024)
    img = normalize_ct(img)

    # load model
    model = heartseg_model.getUnet3d(
        down_steps=4, input_shape=(*input_size, 1), mgpu=1, ext=True
    )
    model.load_weights(model_weights)

    # predict heart segmentation by model
    prd = model.predict(img[np.newaxis, :, :, :, np.newaxis])
    prd = prd[0, :, :, :, 0]

    # recover input shape
    prd = center_crop(prd, bb_shape)
    padding = [
        (bb[0], original_shape[0] - bb[1]),
        (bb[2], original_shape[1] - bb[3]),
        (bb[4], original_shape[2] - bb[5]),
    ]
    prd = np.pad(prd, padding)

    # convert back to SITKImage and recover spacing and size
    prd_sitk = sitk.GetImageFromArray(prd)
    prd_sitk.CopyInformation(img_sitk)
    prd_sitk = change_spacing(prd_sitk, original_spacing, new_size=original_size)
    # remove all regions except the largest one
    prd_sitk = keep_only_the_largest_volume(prd_sitk)
    return prd_sitk


if __name__ == "__main__":
    import argparse

    from util import read_image

    parser = argparse.ArgumentParser(
        description="Perform the second step - an exact heart segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=str, help="the CT image to process")
    parser.add_argument(
        "--heartloc",
        type=str,
        required=True,
        help="nrrd image of the heart localization step",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="the name of the file where the prediction is stored",
    )
    args = parser.parse_args()

    img_sitk = read_image(args.image)
    heartloc_sitk = read_image(args.heartloc) if args.heartloc else None
    prd_sitk = heart_segmentation(img_sitk, heartloc_sitk=heartloc_sitk)

    sitk_writer = sitk.ImageFileWriter()
    sitk_writer.SetUseCompression(True)
    sitk_writer.SetFileName(args.out)
    sitk_writer.Execute(prd_sitk)
