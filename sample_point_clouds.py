import numpy as np

def get_rect(center, width, height, num_points):
    samples = int(num_points/4)
    width_array = np.array([width*np.ones(samples)])
    height_array = np.array([height*np.ones(samples)])
    top = np.hstack((np.array([np.random.uniform(-width, width, samples)]).T, height_array.T))
    bottom = np.hstack((np.array([np.random.uniform(-width, width, samples)]).T, -height_array.T))
    left = np.hstack((width_array.T, np.array([np.random.uniform(-height, height, samples)]).T))
    right = np.hstack((-width_array.T, np.array([np.random.uniform(-height, height, samples)]).T))
    rect = np.vstack((np.vstack((np.vstack((top, bottom)), left)), right))
    rect = rect + center.T
    return rect