import numpy as np
import matplotlib.pyplot as plt
import math

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

def get_rhomb(center, width, height, num_points):
    m = height/width
    x_neg = np.linspace(-width, 0, num_points//4)
    x_pos = np.linspace(0, width, num_points//4)
    y_1 = m*(x_neg+width)
    y_2 = -m*(x_pos-width)
    y_3 = -y_1
    y_4 = -y_2

    s1 = np.hstack((np.array([x_neg]).T, np.array([y_1]).T))
    s2 = np.hstack((np.array([x_neg]).T, np.array([y_3]).T))
    s3 = np.hstack((np.array([x_pos]).T, np.array([y_2]).T))
    s4 = np.hstack((np.array([x_pos]).T, np.array([y_4]).T))

    return np.vstack((s1, s2, s4, s3)) + center


def get_pent(center, side_length, num_points):
    radius = side_length / (2 * np.sin(np.pi / 5))
    angle_step = 2 * np.pi / 5
    coordinates = []
    
    for i in range(5):
        start_angle = i * angle_step
        end_angle = (i + 1) * angle_step
        
        x1 = np.linspace(radius * np.cos(start_angle), radius * np.cos(end_angle), num_points // 5 + 1)
        y1 = np.linspace(radius * np.sin(start_angle), radius * np.sin(end_angle), num_points // 5 + 1)
        
        # Exclude the last point of each segment to avoid duplication at vertices
        if i < 4:
            x1 = x1[:-1]
            y1 = y1[:-1]
        
        coordinates.extend(zip(x1, y1))
    
    coordinates = np.array(coordinates) + center
    
    return np.flip(coordinates)

def get_hex(center, side_length, num_points):
    radius = side_length / (2 * np.sin(np.pi / 6))
    angle_step = 2 * np.pi / 6
    coordinates = []
    
    for i in range(6):
        start_angle = i * angle_step
        end_angle = (i + 1) * angle_step
        
        x1 = np.linspace(radius * np.cos(start_angle), radius * np.cos(end_angle), num_points // 6 + 1)
        y1 = np.linspace(radius * np.sin(start_angle), radius * np.sin(end_angle), num_points // 6 + 1)
        
        # Exclude the last point of each segment to avoid duplication at vertices
        if i < 5:
            x1 = x1[:-1]
            y1 = y1[:-1]
        
        coordinates.extend(zip(x1, y1))
    
    coordinates = np.array(coordinates) + center
    
    return coordinates


def get_triangle(center, base, height, num_points):
    N = num_points//3
    A = np.array([0.0, 0.0])
    B = np.array([base, 0.0])
    C = np.array([base / 2.0, height])
    
    points = []
    # Generate points along the edges
    def add_points_on_edge(p1, p2, num_points):
        for t in np.linspace(0, 1, num_points):
            point = p1 + t * (p2 - p1)
            points.append(point)
    
    add_points_on_edge(A, B, N)
    add_points_on_edge(B, C, N)
    add_points_on_edge(C, A, N)

    return np.array(points) - np.array([base/2, height/2]) + center

'''
rhombus = get_rhomb([0, 0], 1, 1, 100)
plt.plot(rhombus[:,0], rhombus[:,1])
plt.show()
'''

'''
triangle = get_triangle([0, 0], 1, 100)
plt.plot(triangle[:,0], triangle[:,1])
plt.show()
'''
