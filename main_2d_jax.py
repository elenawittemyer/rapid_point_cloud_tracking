import jax
import jaxopt
import jax.numpy as np
import numpy
from jax import jit
from jaxlie import SE2, SE3
from sample_point_clouds import get_rect, get_circle
import time
import scipy

@jit
def sdf_sq(T, point):
    # transform point cloud
    T = SE2.from_xy_theta(T[0], T[1], 0.)
    T_inv = SE2.inverse(T)
    T_p = SE2.from_xy_theta(point[0], point[1], 0.)
    point_pos = np.array([(T_inv@T_p).translation()[0],
                            (T_inv@T_p).translation()[1]])
    
    # measure sdf at each point of transformed point cloud
    a = np.array([0., -1])
    b = np.array([0., 1])
    t = 2.
    p = point_pos
    l = np.linalg.norm(b-a)
    d = (b-a)/l
    q = (p-(a+b)*.5)
    q = np.array([[d[0],-d[1]],
                [d[1],d[0]]])@q
    q = np.abs(q)-np.array([l,t])*0.5
    sdf = np.linalg.norm(np.maximum(q, 0.)) + np.minimum(np.maximum(q[0],q[1]), 0.)
    return sdf


@jit
def sdf_c(T, point):
    # transform point cloud
    T = SE2.from_xy_theta(T[0], T[1], 0.)
    T_inv = SE2.inverse(T)
    T_p = SE2.from_xy_theta(point[0], point[1], 0.)
    point_pos = np.array([(T_inv@T_p).translation()[0],
                          (T_inv@T_p).translation()[1]])
    
    # measure sdf at each point of transformed point cloud
    r = 1.
    sdf = np.linalg.norm(point_pos) - r
    return sdf

batch_sdf_sq = jax.vmap(sdf_sq, in_axes=[None, 0])
batch_sdf_c = jax.vmap(sdf_c, in_axes=[None, 0])

def calc_cost_c(T, point_cloud):
    sdf_array = batch_sdf_c(T, point_cloud)
    cost = np.linalg.norm(np.array(sdf_array), ord=1)
    return cost

def calc_cost_sq(T, point_cloud):
    sdf_array = batch_sdf_sq(T, point_cloud)
    cost = np.linalg.norm(np.array(sdf_array), ord=1)
    return cost

def opt_T(shape, point_cloud):
    if shape == 'circle' or shape == 'Circle' or shape == 'c' or shape == 'C':
        solver = jaxopt.ScipyMinimize(fun = calc_cost_c, maxiter = 500)
        T_opt, state = solver.run(np.array([0., 0.]), point_cloud)
    elif shape == 'square' or shape == 'Square'  or shape == 'sq' or shape == 'Sq':
        solver = jaxopt.ScipyMinimize(fun = calc_cost_sq, maxiter = 500)
        T_opt, state = solver.run(np.array([0., 0.]), point_cloud)
    else:
        return 'unrecognized shape'
    
    return T_opt, state.fun_val

def assign_primitive(shapes, point_cloud):
    sdf_min = opt_T(shapes[0], point_cloud)
    assigned_shape = shapes[0]
    for i in range(1, len(shapes)):
        sdf_current = opt_T(shapes[i], point_cloud)
        if sdf_current[1] < sdf_min[1]:
            sdf_min = sdf_current
            assigned_shape = shapes[i]
    return sdf_min[0], sdf_min[1], assigned_shape



#point_cloud_test = get_circle(np.array([2, 2]), 1, 1000)
point_cloud_test = get_rect(np.array([2,2]), 1, 1, 1000)
test_shapes = ['square', 'circle']
start_time = time.time()
print(assign_primitive(test_shapes, point_cloud_test))
print(time.time()-start_time)


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

