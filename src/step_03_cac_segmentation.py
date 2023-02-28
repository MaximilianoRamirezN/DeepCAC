from typing import List
import numpy as np
import SimpleITK as sitk

SITKImage = sitk.SimpleITK.Image

import step3_cacseg.cacseg_model as cacseg_model

from util import (
    normalize_ct,
    change_spacing,
    get_bounding_box,
    keep_only_the_largest_volume,
    dilate,
    pack_into_cubes,
    unpack_from_cubes,
    get_ag,
    get_ag_class,
    center_crop,
    center_pad,
)


def cac_segmentation(
    img_sitk: SITKImage,
    heartseg_sitk: SITKImage,
    input_spacing: List[int] = [0.68, 0.68, 2.5],
    cube_size: List[int] = [32, 64, 64],
    model_weights: str = "data/step3_cacseg/model_weights/step3_cacseg_model_weights.hdf5",
):
    """
    Perform the third step of the DeepCAC pipeline - CAC segmentation.

    Parameters
    ----------
    img_sitk: SITKImage
        the CT image to be processed
    heartseg_sitk: SITKImage
        the heart segmentation mask of the second step of the DeepCAC pipeline
    input_spacing: List[int], optional
        the CT image is resampled to this spacing before processing with the
        neural network
    cube_size: List[int], optional
        the image is processed as non-overlapping cubes of this size by the model
    model_weights: str, optional
        filename of the model weights for the neural network

    Returns
    -------
    SITKImage
        a binary segmentation mask of the CAC with the same
        spacing and size as the input image
    """
    # change spacing of segmentation mask and dilate
    heartseg_sitk = change_spacing(heartseg_sitk, input_spacing)
    heartseg_sitk = dilate(heartseg_sitk)
    heartseg = sitk.GetArrayFromImage(heartseg_sitk)

    # change spacing of the image
    original_spacing = img_sitk.GetSpacing()
    original_size = img_sitk.GetSize()
    img_sitk = change_spacing(img_sitk, input_spacing)
    img = sitk.GetArrayFromImage(img_sitk)
    original_shape = img.shape

    # set all non-segmented pixels to air
    img[heartseg == 0] = -1024
    # select the bounding box from the image and normalize
    bb = get_bounding_box(heartseg)
    img = img[bb[0] : bb[1], bb[2] : bb[3], bb[4] : bb[5]]
    img = normalize_ct(img)

    # load model
    model = cacseg_model.getUnet3d(
        down_steps=3,
        input_shape=(*cube_size, 1),
        pool_size=(2, 2, 2),
        conv_size=(3, 3, 3),
        initial_learning_rate=0.0001,
        mgpu=1,
        extended=True,
        drop_out=0.5,
        optimizer="ADAM",
    )
    model.load_weights(model_weights)

    # pack image, predict with model and unpack prediction
    img_cubes, n_cubes = pack_into_cubes(img, cube_size)
    prd_cubes = model(img_cubes[:, :, :, :, np.newaxis])[:, :, :, :, 0]
    prd = unpack_from_cubes(prd_cubes, n_cubes, img.shape)

    # recover input shape
    padding = (
        (bb[0], original_shape[0] - bb[1]),
        (bb[2], original_shape[1] - bb[3]),
        (bb[4], original_shape[2] - bb[5]),
    )
    prd = np.pad(prd, padding)

    # load as SITK image and recover input spacing and size
    prd_sitk = sitk.GetImageFromArray(prd)
    prd_sitk.CopyInformation(img_sitk)
    prd_sitk = change_spacing(prd_sitk, original_spacing, new_size=original_size)
    return prd_sitk


if __name__ == "__main__":
    import argparse

    from util import read_image

    parser = argparse.ArgumentParser(
        description="Perform the third step - CAC segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=str, help="the CT image to process")
    parser.add_argument(
        "--heartseg",
        type=str,
        required=True,
        help="nrrd image of the heart segmentation step",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="the name of the file where the prediction is stored",
    )
    args = parser.parse_args()

    img_sitk = read_image(args.image)
    heartseg_sitk = read_image(args.heartseg) if args.heartseg else None
    prd_sitk = cac_segmentation(img_sitk, heartseg_sitk=heartseg_sitk)

    sitk_writer = sitk.ImageFileWriter()
    sitk_writer.SetUseCompression(True)
    sitk_writer.SetFileName(args.out)
    sitk_writer.Execute(prd_sitk)
