import jax.numpy as np
import numpy
rng = numpy.random.default_rng()
from jax import jacfwd, grad, jit, vmap, hessian
from jaxlie import SE2, SE3
from primitives import sdfBox, sdfCircle
from sample_point_clouds import get_rect, get_circle
import time
import scipy


def usdf_sq(T, points):
    # transform point cloud
    T = SE2.from_xy_theta(T[0], T[1], 0.)
    T_inv = SE2.inverse(T)
    transformed_point_cloud = []
    for i in range(len(points)):
        T_p = SE2.from_xy_theta(points[i][0], points[i][1], 0.)
        point_pos = np.array([(T_inv@T_p).translation()[0],
                              (T_inv@T_p).translation()[1]])
        transformed_point_cloud.append(point_pos)
    transformed_point_cloud = np.array(transformed_point_cloud)
    
    # measure sdf at each point of transformed point cloud
    sdf_array = []
    a = np.array([0., -1])
    b = np.array([0., 1])
    t = 2.
    for i in range(len(transformed_point_cloud)):
        p = transformed_point_cloud[i]
        l = np.linalg.norm(b-a)
        d = (b-a)/l
        q = (p-(a+b)*.5)
        q = np.array([[d[0],-d[1]],
                    [d[1],d[0]]])@q
        q = np.abs(q)-np.array([l,t])*0.5
        sdf = np.linalg.norm(np.maximum(q, 0.)) + min(max(q[0],q[1]), 0.)
        sdf_array.append(sdf)

    # find sdf cost
    cost = np.linalg.norm(np.array(sdf_array), ord=1)
    return cost

def usdf_c(T, points):
    # transform point cloud
    T = SE2.from_xy_theta(T[0], T[1], 0.)
    T_inv = SE2.inverse(T)
    transformed_point_cloud = []
    for i in range(len(points)):
        T_p = SE2.from_xy_theta(points[i][0], points[i][1], 0.)
        point_pos = np.array([(T_inv@T_p).translation()[0],
                              (T_inv@T_p).translation()[1]])
        transformed_point_cloud.append(point_pos)
    transformed_point_cloud = np.array(transformed_point_cloud)
    
    # measure sdf at each point of transformed point cloud
    sdf_array = []
    r = 1.
    for i in range(len(transformed_point_cloud)):
        sdf = np.linalg.norm(transformed_point_cloud[i]) - r
        sdf_array.append(sdf)

    # find sdf cost
    cost = np.linalg.norm(np.array(sdf_array), ord=1)
    return cost

sdf_c_iter = []
sdf_sq_iter = []
Nfeval_c = 1
Nfeval_sq = 1

def callbackC(sdf):
    global Nfeval_c
    sdf_c_iter.append(sdf)
    Nfeval_c += 1

def callbackSq(sdf):
    global Nfeval_sq
    sdf_sq_iter.append(sdf)
    Nfeval_sq += 1


def opt_T(shape, point_cloud):
    if shape == 'circle' or shape == 'Circle' or shape == 'c' or shape == 'C':
        T_opt = scipy.optimize.minimize(usdf_c, x0 = np.array([0., 0.]), args = point_cloud,
                                        method = 'Nelder-Mead', tol = 1e-3) # add callback = callbackC if you want iterations
    elif shape == 'square' or shape == 'Square'  or shape == 'sq' or shape == 'Sq':
        T_opt = scipy.optimize.minimize(usdf_sq, x0 = np.array([0., 0.]), args = point_cloud,
                                     method = 'Nelder-Mead', tol = 1e-3) # add callback = callbackSq if you want iterations
    else:
        return 'unrecognized shape'
    
    return T_opt
    
def assign_primitive(shapes, point_cloud):
    sample_indices = rng.choice(len(point_cloud), 30, replace=False)
    point_cloud_sample = point_cloud[sample_indices]

    sdf_min = opt_T(shapes[0], point_cloud_sample)
    assigned_shape = shapes[0]
    for i in range(1, len(shapes)):
        sdf_current = opt_T(shapes[i], point_cloud_sample)
        if sdf_current.fun < sdf_min.fun:
            sdf_min = sdf_current
            assigned_shape = shapes[i]
    return sdf_min.x, assigned_shape

point_cloud_test = get_rect(np.array([2,2]), 1, 1, 1000)
#point_cloud_test = get_circle(np.array([2, 2]), 1, 100000)
print(assign_primitive(['c', 'sq'], point_cloud_test))
#sample_indices = rng.choice(len(point_cloud_test), 30, replace=False)
#point_cloud_sample = point_cloud_test[sample_indices]





################################
## visualization helpers #######
################################
'''
with open('Visualization/sq_iter.txt', 'w') as f:
    for line in sdf_sq_iter:
        f.write(f"{line}\n")

with open('Visualization/c_iter.txt', 'w') as f:
    for line in sdf_c_iter:
        f.write(f"{line}\n")
'''

