from typing import List

import numpy as np
import SimpleITK as sitk

SITKImage = sitk.SimpleITK.Image

import step1_heartloc.heartloc_model as heartloc_model
from util import (
    read_image,
    normalize_ct,
    change_spacing,
    center_crop,
    center_pad,
    keep_only_the_largest_volume,
)


def heart_localization(
    img_sitk: SITKImage,
    input_spacing: List[int] = [3.0, 3.0, 3.0],
    input_size: List[int] = [112, 112, 112],
    model_weights: str = "data/step1_heartloc/model_weights/step1_heartloc_model_weights.hdf5",
) -> SITKImage:
    """
    Perform the first step of the DeepCAC pipeline - a rough heart segmentation.

    Parameters
    ----------
    img_sitk: SITKImage
        the CT image to be processed
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
        a rough binary segmentation mask of the heart with the same
        spacing and size as the input image
    """
    # save original spacing of the input
    original_spacing = img_sitk.GetSpacing()
    original_size = img_sitk.GetSize()

    # change spacing for model
    img_sitk = change_spacing(img_sitk, input_spacing)
    # get as array and save original shape
    img = sitk.GetArrayFromImage(img_sitk)
    original_shape = img.shape

    # crop and pad to math model expected input
    img = center_crop(img, input_size)
    img = center_pad(img, input_size, constant=-1024)
    # normalize
    img = normalize_ct(img)

    # load model
    model = heartloc_model.get_unet_3d(
        down_steps=4, input_shape=(*input_size, 1), mgpu=1, ext=False
    )
    model.load_weights(model_weights)

    # compute prediction
    prd = model.predict(img[np.newaxis, :, :, :, np.newaxis])
    prd = prd[0, :, :, :, 0]
    # pad and crop to original_shape shape
    prd = center_pad(prd, original_shape)
    prd = center_crop(prd, original_shape)

    # convert to sitk image
    prd_sitk = sitk.GetImageFromArray(prd)
    prd_sitk.CopyInformation(img_sitk)
    # change spacing to restore original spacing and resolution
    prd_sitk = change_spacing(prd_sitk, original_spacing, new_size=original_size)
    # remove all but the largest predicted region
    prd_sitk = keep_only_the_largest_volume(prd_sitk)
    return prd_sitk


if __name__ == "__main__":
    import argparse

    from util import read_image

    parser = argparse.ArgumentParser(
        description="Perform the first step - a rough heart localization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=str, help="the CT image to process")
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="the name of the file where the prediction is stored",
    )
    args = parser.parse_args()

    img_sitk = read_image(args.image)
    prd_sitk = heart_localization(img_sitk)

    sitk_writer = sitk.ImageFileWriter()
    sitk_writer.SetUseCompression(True)
    sitk_writer.SetFileName(args.out)
    sitk_writer.Execute(prd_sitk)
