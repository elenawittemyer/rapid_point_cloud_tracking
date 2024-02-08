import numpy as np

def get_cube(length, density):
    max_samples = length*100000
    samples = int(density*max_samples)
    flat_cube = np.random.uniform(0., length, 3*samples)
    cube = np.reshape(flat_cube, (samples, 3))
    return cube