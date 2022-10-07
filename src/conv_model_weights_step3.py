import os

import h5py
import numpy as np
from tensorflow.keras.layers import Conv3D, BatchNormalization
import yaml

import step3_cacseg.cacseg_model as cacseg_model

weights_file = "../data/step3_cacseg/model_weights/step3_cacseg_model_weights.hdf5"

mgpu = 1
down_steps = 3  
cube_size = [64, 64, 32]
input_shape = (cube_size[2], cube_size[1], cube_size[0], 1)
pool_size = (2, 2, 2)
conv_size = (3, 3, 3)
lr = 0.0001
extended = True
drop_out = 0.5
optimizer = 'ADAM'
model = cacseg_model.getUnet3d(down_steps = down_steps, input_shape = input_shape, pool_size = pool_size,
                                 conv_size = conv_size, initial_learning_rate = lr, mgpu = mgpu,
                                 extended = extended, drop_out = drop_out, optimizer = optimizer)

weights = h5py.File(weights_file, mode="r")
weights = weights["model_1"]

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

model.save(weights_file, save_format="h5")
