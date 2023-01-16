import argparse

import cv2 as cv
import numpy as np
import SimpleITK as sitk

from util import read_image, window

parser = argparse.ArgumentParser(
    description="Display a CT image and a prediction/segmentation mask",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--img", type=str, required=True, help="the ct image")
parser.add_argument("--seg", type=str, required=True, help="the segmentation mask")
args = parser.parse_args()

# load images
img_sitk = read_image(args.img)
img = sitk.GetArrayFromImage(img_sitk)

prd_sitk = read_image(args.seg)
prd = sitk.GetArrayFromImage(prd_sitk)

# normalize
img = window(img)
img = (img - img.min()) / (img.max() - img.min())
prd = (prd - prd.min()) / (prd.max() - prd.min())

# display loop
i = 0
timeout = 100
while True:
    slice_img = img[i]
    slice_img = (slice_img * 255).astype(np.uint8)
    slice_img = cv.resize(slice_img, (512, 512))

    slice_prd = prd[i]
    slice_prd = (slice_prd * 255).astype(np.uint8)
    slice_prd = cv.resize(slice_prd, (512, 512))

    i = (i + 1) % img.shape[0]

    cnts, _ = cv.findContours(slice_prd, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    slice_img = slice_img.repeat(3).reshape(*slice_img.shape, 3)
    slice_com = cv.drawContours(slice_img.copy(), cnts, -1, (255, 127, 0), -1)
    slice_prd = slice_prd.repeat(3).reshape(*slice_prd.shape, 3)

    slice_txt = f"{str(i):>{len(str(img.shape[0]))}}/{img.shape[0]}"
    slice_img = cv.putText(
        slice_img, slice_txt, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3
    )

    space = np.full((slice_img.shape[0], 10, 3), 239, np.uint8)
    cv.imshow(
        "Segmentation", np.hstack((slice_img, space, slice_com, space, slice_prd))
    )
    key = cv.waitKey(timeout)
    if key == ord("q"):
        break
    elif key == ord("p"):
        timeout = 0 if timeout > 0 else 100
    elif key == ord("b") or key == 81:
        i = (i - 2) % img.shape[0]
