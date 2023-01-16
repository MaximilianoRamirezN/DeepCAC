import math
import os
from typing import Optional, List, Tuple

import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk

SITKImage = sitk.SimpleITK.Image


def read_image(filename: str) -> SITKImage:
    """
    Read an image with SITK either from an image file or a series.

    Parameters
    ----------
    filename: str
        The file from which the image is read. If it is a directory
        it is treated as a series, else as a file.

    Returns
    -------
    SITKImage
    """
    if os.path.isdir(filename):
        sitk_reader = sitk.ImageSeriesReader()
        sitk_reader.SetFileNames(sitk_reader.GetGDCMSeriesFileNames(filename))
    else:
        sitk_reader = sitk.ImageFileReader()
        sitk_reader.SetFileName(filename)
    return sitk_reader.Execute()


def normalize_ct(ct: np.ndarray) -> np.ndarray:
    """
    Normalize a CT image for processing by a neural network.

    Afterwards the value range of the CT is [-1, 1].

    Parameters
    ----------
    ct: np.ndarray

    Returns
    -------
    np.ndarray
    """
    return (np.clip(ct, -1024, 3071) - 1023.5) / 2047.5


def change_spacing(
    img: SITKImage,
    new_spacing: Tuple[int, int, int],
    new_size: Optional[Tuple[int, int, int]] = None,
) -> SITKImage:
    """
    Change the spacing and size of an SITKImage.

    Parameters
    ----------
    img: SITKImage
        The image of which the spacing and size is changed.
    new_spacing: Tuple[int, int, int]
        The new spacing.
    new_size: Tuple[int, int, int], optional
        The new size which is computed from the old spacing and size
        if not provided.

    Returns
    -------
    SITKImage
    """
    old_size = np.array(img.GetSize())
    old_spacing = np.array(img.GetSpacing())

    if new_size is None:
        new_size = (old_size * old_spacing) / np.array(new_spacing)
        new_size = tuple(map(round, new_size))

    return sitk.Resample(
        img, new_size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(), new_spacing
    )


def keep_only_the_largest_volume(
    pred_sitk: SITKImage, threshold: float = 0.5
) -> SITKImage:
    """
    Remove all volumes except the largest one from a segmentation.

    Parameters
    ----------
    pred_sitk: SITKImage
        the segmentation mask
    threshold: float, optional
        the threshold applied to the segmentation to ensure that it
        is a binary mask

    Returns
    -------
    SITKImage
    """
    strucuring_elem = [
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ]

    pred = sitk.GetArrayFromImage(pred_sitk)
    pred[pred <= threshold] = 0
    pred[pred > 0] = 1

    label_img, num_labels = ndimage.label(pred, structure=strucuring_elem)

    if num_labels == 0:
        return pred_sitk

    volume_sizes = [
        np.sum(label_img == label_num) for label_num in range(1, num_labels + 1)
    ]
    max_label = np.argmax(volume_sizes) + 1

    pred = np.zeros(pred.shape)
    pred[label_img == max_label] = 1
    _pred_sitk = sitk.GetImageFromArray(pred)
    _pred_sitk.CopyInformation(pred_sitk)
    return _pred_sitk


def dilate(
    pred_sitk: SITKImage, iterations: int = 11, threshold: float = 0.9
) -> SITKImage:
    """
    Dilate a segmentation mask.

    Parameters
    ----------
    pred_sitk: SITKImage
        the segmentation mask
    iterations: int, optional
        number of iterations of dilations
    threshold: float, optional
        threshold applied to ensure that the segmentation mask is
        binary

    Returns
    -------
    SITKImage
    """
    strucuring_elem = np.zeros((3, 3, 3))
    strucuring_elem[0] = [
        [False, False, False],
        [False, True, False],
        [False, False, False],
    ]
    strucuring_elem[1] = [
        [False, True, False],
        [True, True, True],
        [False, True, False],
    ]
    strucuring_elem[2] = [
        [False, False, False],
        [False, True, False],
        [False, False, False],
    ]

    pred = sitk.GetArrayFromImage(pred_sitk)
    pred[pred <= threshold] = 0
    pred[pred > 0] = 1
    pred = ndimage.binary_dilation(
        pred, structure=strucuring_elem, iterations=iterations
    ).astype(pred.dtype)

    _pred_sitk = sitk.GetImageFromArray(pred)
    _pred_sitk.CopyInformation(pred_sitk)
    return _pred_sitk


def get_bounding_box(
    pred: np.ndarray, min_size: Tuple[int, int, int] = (32, 48, 48)
) -> List[int]:
    """
    Compute the bounding box of the segmentation mask.

    Parameters
    ----------
    pred: np.ndarray
        the segmentation mask
    min_size: Tuple[int, int, int], optional
        The minimal size of the return bounding box.
        This means the bounding box is enlarged if it is smaller.

    Parameters
    ----------
    List[int]
        a boundinx box as [z_min, z_max, y_min, y_max, x_min, x_max]
    """
    zs, ys, xs = np.where(pred > 0)

    bb = [zs.min(), zs.max(), ys.min(), ys.max(), xs.min(), xs.max()]
    if (bb[1] - bb[0]) < min_size[0]:
        offset = math.ceil((min_size[0] - (bb[1] - bb[0])) / 2)
        bb[0] = max(0, bb[0] - offset)
        bb[1] = bb[0] + min_size[0]

    if (bb[3] - bb[2]) < min_size[1]:
        offset = math.ceil((min_size[1] - (bb[3] - bb[2])) / 2)
        bb[2] = max(0, bb[2] - offset)
        bb[3] = bb[2] + min_size[1]

    if (bb[5] - bb[4]) < min_size[2]:
        offset = math.ceil((min_size[2] - (bb[5] - bb[4])) / 2)
        bb[4] = max(0, bb[4] - offset)
        bb[5] = bb[4] + min_size[2]
    return bb


def pack_into_cubes(
    img: np.ndarray, cube_size: List[int]
) -> Tuple[np.ndarray, List[int]]:
    """
    Pack an image as cubes.

    This means that the image is divided into non-overlapping cubes and repacked
    so that the first dimension of the resulting image is the cube index.

    Parameters
    ----------
    img: np.ndarray
        the image to be packed
    cube_size: List[int]
        the size of the cubes

    Returns
    -------
    np.ndarray
        an image with shape [n_cubes, *cube_size]
    n_cubes
        the number of cubes created for each dimension [n_cubes_z, n_cubes_y, n_cubes_x]
    """
    n_cubes = np.ceil(np.array(img.shape, dtype=float) / cube_size).astype(int)
    size_new = list(n_cubes * cube_size)

    img_padded = np.zeros(size_new)
    img_padded[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img

    _n_cubes = np.prod(n_cubes)
    _n_cubes = np.ceil(_n_cubes / 4).astype(int) * 4

    img_cubes = np.full((_n_cubes, *cube_size), -1, dtype=np.float64)  # -1 = air

    count = 0
    for z in range(n_cubes[0]):
        for y in range(n_cubes[1]):
            for x in range(n_cubes[2]):
                img_cubes[count] = img_padded[
                    z * cube_size[0] : (z + 1) * cube_size[0],
                    y * cube_size[1] : (y + 1) * cube_size[1],
                    x * cube_size[2] : (x + 1) * cube_size[2],
                ]
                count += 1

    return img_cubes, n_cubes


def unpack_from_cubes(
    img_cubes: np.ndarray, n_cubes: List[int], img_shape: List[int]
) -> np.ndarray:
    """
    Unpack an image from cubes.

    This reverts the operation performed with `pack_into_cubes`.

    Parameters
    ----------
    img_cubes: np.ndarray
        cube image as returned by `pack_into_cubes`
    n_cubes: List[int]
        number of cubes in each dimension as returned by `pack_into_cubes`
    img_shape:
        The shape of the image input into `pack_into_cubes`.
        This cannot be recovered automatically.

    Returns
    -------
    np.ndarray
    """
    cube_size = np.array(img_cubes.shape[1:])

    size_padded = list(np.array(n_cubes) * cube_size)
    img_padded = np.zeros(size_padded)

    count = 0
    for z in range(n_cubes[0]):
        for y in range(n_cubes[1]):
            for x in range(n_cubes[2]):
                img_padded[
                    z * cube_size[0] : (z + 1) * cube_size[0],
                    y * cube_size[1] : (y + 1) * cube_size[1],
                    x * cube_size[2] : (x + 1) * cube_size[2],
                ] = img_cubes[count]
                count += 1

    return img_padded[: img_shape[0], : img_shape[1], : img_shape[2]]


def get_object_ag(clc_object: np.ndarray, object_volume: float):
    """
    Compute the Agatston-Score of a 2D region.

    Parameters
    ----------
    clc_object: np.ndarray
        an array of pixel values in the CT image belonging to the region
    object_volume: float
        the volume of the object (I guess in cm^3)

    Returns
    -------
    float
    """
    object_max = np.max(clc_object)
    if 130 <= object_max < 200:
        return object_volume * 1
    if 200 <= object_max < 300:
        return object_volume * 2
    if 300 <= object_max < 400:
        return object_volume * 3
    else:
        return object_volume * 4


def get_ag(
    img: np.ndarray,
    msk: np.ndarray,
    spacing: List[int],
    min_n_pixels: int = 3,
    ag_div: int = 3,
):
    """
    Compute the Agatston-Score of an image.

    Parameters
    ----------
    img: np.ndarray
        the CT image
    msk: np.ndarray
        binary segmentation mask of the calcium in the arteries
    spacing: List[int]
        the spacing of pixels in the image which is used to
        compute the volume of a pixel
    min_n_pixels: int, optional
        regions smaller than this in a slice are ignored
    ag_div: int, optional
        divider for the pixel volume

    Returns
    -------
    float
    """
    px_volume = np.array(spacing).prod() / ag_div

    ag = 0
    # Loop over all slices:
    for img_slice, msk_slice in zip(img, msk):

        # Get all objects in mask
        label_img, num_labels = ndimage.label(msk_slice, structure=np.ones((3, 3)))

        # Process each object
        for label in range(1, num_labels + 1):
            obj_msk = label_img == label
            n_pixels = obj_msk.sum()

            # 1) Remove small objects
            if n_pixels <= min_n_pixels:
                continue
            # 2) Calculate volume
            object_volume = n_pixels * px_volume
            # 3) Calculate AG for object
            object_ag = get_object_ag(img_slice[obj_msk], object_volume)
            # 4) Sum up scores
            ag += object_ag
    return ag


def get_ag_class(ag: float) -> int:
    """
    Get the risk class based on the Agatston-Score.

    Parameters
    ----------
    ag: float
        the Agatston-Score

    Returns
    -------
    int
    """
    ag = round(ag, 3)
    if ag == 0:
        return 0
    elif 0 < ag <= 100:
        return 1
    elif 100 < ag <= 300:
        return 2
    else:
        return 3


def center_crop(img: np.ndarray, target_shape: List[int]) -> np.ndarray:
    """
    Crop the center of an image.

    Parameters
    ----------
    img: np.ndarray
        the image to be cropped
    target_shape: List[int]
        the shape of the resulting crop

    Returns
    -------
    np.ndarray
    """
    assert len(img.shape) == len(target_shape)

    slices = []
    for dim_i, dim_t in zip(img.shape, target_shape):
        if dim_t >= dim_i:
            _slice = slice(0, dim_i)
        else:
            diff_h = (dim_i - dim_t) / 2
            _slice = slice(math.ceil(diff_h), dim_i - math.floor(diff_h))
        slices.append(_slice)
    slices = tuple(slices)
    return img[slices]


def center_pad(
    img: np.ndarray, target_shape: Tuple[int, int, int], constant: float = 0
) -> np.ndarray:
    """
    Pad an image so that it is in the center of the result.

    Parameters
    ----------
    img: np.ndarray
        the image to be padded
    target_shape: Tuple[int, int, int]
        the shape of the image after padding
    constant: float, optional
        the constant with which the image is padded

    Returns
    -------
    np.ndarray
    """
    assert len(img.shape) == len(target_shape)

    padding = []
    for dim_i, dim_t in zip(img.shape, target_shape):
        if dim_t <= dim_i:
            _padding = (0, 0)
        else:
            diff_h = (dim_t - dim_i) / 2
            _padding = (math.ceil(diff_h), math.floor(diff_h))
        padding.append(_padding)
    return np.pad(img, padding, mode="constant", constant_values=constant)
