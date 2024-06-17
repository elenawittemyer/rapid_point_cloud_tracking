import numpy as np
import matplotlib.pyplot as plt

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

def get_triangle(center, l, num_points):
    r = l/np.cos(np.pi/6)
    samples = num_points//3
    a = center[0]
    b = center[1]
    v1 = np.array([a, b+r])
    v2 = np.array([a+r*np.cos(np.pi/6), b-r*np.sin(np.pi/6)])
    v3 = np.array([a-r*np.cos(np.pi/6), b-r*np.sin(np.pi/6)])
    
    l23 = np.vstack((np.linspace(v3[0], v2[0], samples), np.linspace(v3[1], v2[1], samples))).T
    l12 = np.vstack((np.linspace(v1[0], v2[0], samples), np.linspace(v1[1], v2[1], samples))).T
    l31 = np.vstack((np.linspace(v3[0], v1[0], samples), np.linspace(v3[1], v1[1], samples))).T

    return np.vstack((np.vstack((l23, l12)), l31))

'''
triangle = get_triangle([0,0], 1, 100)
plt.plot(triangle[:,0], triangle[:,1])
plt.show()
'''