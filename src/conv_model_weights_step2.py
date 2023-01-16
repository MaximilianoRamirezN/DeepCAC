import os

import h5py
import numpy as np
from tensorflow.keras.layers import Conv3D, BatchNormalization
import yaml

import step2_heartseg.heartseg_model as heartseg_model

filename_weights = "../data/step2_heartseg/model_weights/step2_heartseg_model_weights.hdf5"

conf = "./step2_heart_segmentation.yaml"
base_conf_file_path = 'config/'

conf_file_path = os.path.join(base_conf_file_path, conf)
with open(conf_file_path) as f:
  yaml_conf = yaml.load(f, Loader = yaml.FullLoader)


down_steps = yaml_conf["model"]["down_steps"]
training_size = yaml_conf["processing"]["training_size"]


mgpu = 1
inputShape = (training_size[2], training_size[1], training_size[0], 1)
model = heartseg_model.getUnet3d(down_steps=down_steps, input_shape=inputShape, mgpu=mgpu, ext=True)


file_weights = h5py.File(filename_weights, mode="r")
weights = file_weights["model_1"]

def set_conv_weights(layer, weights):
    bias = np.array(weights["bias:0"])
    kernel = np.array(weights["kernel:0"])
    layer.set_weights([kernel, bias])

def set_batch_norm_weights(layer, weights):
    beta = np.array(weights["beta:0"])
    gamma = np.array(weights["gamma:0"])
    moving_mean = np.array(weights["moving_mean:0"])
    moving_var = np.array(weights["moving_variance:0"])
    layer.set_weights([gamma, beta, moving_mean, moving_var])


set_weights = {
    Conv3D: set_conv_weights,
    BatchNormalization: set_batch_norm_weights
}


for layer in model.layers:
    name = layer.name
    print(f"Set weights of layer {name} ...", end="\r")
    if name not in weights.keys():
        print(f"Set weights of layer {name} ... skip because weights could not be found")
        continue

    if type(layer) not in set_weights:
        print(f"Set weights of layer {name} ... skip because layers of type {type(layer)} are not supported")

    layer_weights = weights[name]
    set_weights[type(layer)](layer, layer_weights)

file_weights.close()
model.save(filename_weights, save_format="h5")
