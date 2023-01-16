import argparse
import os

import SimpleITK as sitk

from util import read_image

from step_01_heart_localization import heart_localization
from step_02_heart_segmentation import heart_segmentation
from step_03_cac_segmentation import cac_segmentation
from step_04_cac_scoring import cac_scoring

parser = argparse.ArgumentParser(
    description="Perform the entire DeepCAC pipeline to compute the Agatston-Score for a CT image",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("image", type=str, help="the CT image to be processed")
parser.add_argument(
    "--out",
    type=str,
    help="optional output directory where segmentation masks are saved",
)
parser.add_argument("--prefix", type=str, help="prefix for files written")
args = parser.parse_args()
args.prefix = f"{args.prefix}_" if args.prefix else ""

img_sitk = read_image(args.image)

print(f"1 - Perform heart localization ...")
heartloc_sitk = heart_localization(img_sitk)
print(f"2 - Perform heart segmentation ...")
heartseg_sitk = heart_segmentation(img_sitk, heartloc_sitk)
print(f"3 - Perform CAC segmentation ...")
cacseg_sitk = cac_segmentation(img_sitk, heartseg_sitk)
print(f"4 - Compute score ...")
ag, ag_class = cac_scoring(img_sitk, cacseg_sitk)

print(f"Agatson score is {ag:.3f} which is class {ag_class}")

if args.out:
    sitk_writer = sitk.ImageFileWriter()
    sitk_writer.SetUseCompression(True)
    for filename, sitk in [
        ("heartloc", heartloc_sitk),
        ("heartseg", heartseg_sitk),
        ("cacseg", cacseg_sitk),
    ]:
        filename = os.path.join(args.out, f"{args.prefix}{filename}.nrrd")
        sitk_writer.SetFileName(filename)
        sitk_writer.Execute(sitk)
