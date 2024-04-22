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



def get_circle(center, radius, num_points):
    x_point = np.random.uniform(0, radius, int(num_points/4))
    x_point = np.concatenate((x_point, -1*x_point))
    y_point = np.sqrt(1-x_point**2)
    pos_circle = np.vstack((x_point, y_point)).T
    neg_y_point = -1*y_point
    neg_circle = np.vstack((x_point, neg_y_point)).T
    circle = np.vstack((pos_circle, neg_circle))
    circle = circle + center
    return circle