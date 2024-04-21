import jax.numpy as np
import os
wd = os.getcwd()

point_clouds = np.load('Visualization/point_clouds.npy')
point_cloud_pos = np.load('Visualization/point_cloud_pos.npy')
sdfs = np.load('Visualization/sdf_guess.npy')
transforms = np.load('Visualization/transform_guess.npy')
twists = np.load('Visualization/twist_guess.npy')

#TODO: fix velocities