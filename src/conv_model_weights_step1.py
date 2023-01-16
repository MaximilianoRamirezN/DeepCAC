import os

import h5py
import numpy as np
from tensorflow.keras.layers import Conv3D
import yaml

import step1_heartloc.heartloc_model as heartloc_model

conf = "step1_heart_localization.yaml"
base_conf_file_path = 'config/'

conf_file_path = os.path.join(base_conf_file_path, conf)
with open(conf_file_path) as f:
  yaml_conf = yaml.load(f, Loader = yaml.FullLoader)

down_steps = yaml_conf["model"]["down_steps"]
model_input_size = yaml_conf["processing"]["model_input_size"]
extended = yaml_conf["model"]["extended"]


mgpu = 1
crop_size = model_input_size
input_shape = (crop_size, crop_size, crop_size, 1)
model = heartloc_model.get_unet_3d(down_steps = down_steps, input_shape = input_shape, mgpu = mgpu, ext = extended)


filename_weights = "../data/step1_heartloc/model_weights/step1_heartloc_model_weights.hdf5"
file_weights = h5py.File(filename_weights, mode="r")
weights = file_weights["model_1"]

for layer in model.layers:
    name = layer.name
    print(f"Set weights of layer {name} ...", end="\r")
    if name not in weights.keys():
        print(f"Set weights of layer {name} ... skip because weights could not be found")
        continue

    if type(layer) != Conv3D:
        print(f"Set weights of layer {name} ... skip because layers of type {type(layer)} are not supported")
        continue

    layer_weights = weights[name]

    bias = np.array(layer_weights["bias:0"])
    kernel = np.array(layer_weights["kernel:0"])
    layer.set_weights([kernel, bias])

file_weights.close()

model.save(filename_weights, save_format="h5")
