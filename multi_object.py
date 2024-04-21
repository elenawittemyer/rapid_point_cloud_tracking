import jax
import jaxopt
import jax.numpy as np
import numpy
from jax import jit
from functools import partial
import jaxlie
from jaxlie import SE2, SE3
from sample_point_clouds import get_rect, get_circle
import time
import scipy

global_num_clouds = 5
global_num_shapes = 2

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

def calc_cost_c(args, point_clouds):
    N = args["N"]
    T = args["T"]

    cost = 0
    seg_clouds = cloud_splitting(N, point_clouds)
    for i in range(len(seg_clouds)):
        sdf_array = batch_sdf_c(T[i], seg_clouds[i])
        cost += np.linalg.norm(np.array(sdf_array), ord=1)
    return cost

#TODO: change T to be an array of Ts of length N

def calc_cost_sq(args, point_clouds):
    N = args["N"]
    T = args["T"]
    seg_clouds = cloud_splitting(N, point_clouds)
    cost = 0
    for i in range(len(seg_clouds)):
        sdf_array = batch_sdf_sq(T[i], seg_clouds[i])
        cost += np.linalg.norm(np.array(sdf_array), ord=1)
    return cost

#calc_cost_sq = jit(calc_cost_sq, static_argnums = (0,))
#calc_cost_c = jit(calc_cost_c, static_argnums = (0,))

#TODO: try LBFGS for optimization

def opt_T(shape, point_clouds):

    args = {
        "N" : 1.0,
        "T" : np.zeros((1, 2))
    }

    if shape == 'circle' or shape == 'Circle' or shape == 'c' or shape == 'C':
        solver = jaxopt.ScipyMinimize(method = 'Nelder-Mead', fun = calc_cost_c, maxiter = 500, jit=False)
        arg_opt, state = solver.run(args, point_clouds)
    elif shape == 'square' or shape == 'Square'  or shape == 'sq' or shape == 'Sq':
        solver = jaxopt.ScipyMinimize(method = 'Nelder-Mead', fun = calc_cost_sq, maxiter = 500, jit=False)
        arg_opt, state = solver.run(args, point_clouds)
    else:
        return 'unrecognized shape'
    
    return arg_opt, state.fun_val

def cloud_splitting(num_clouds, point_clouds):
    divis_clouds = (np.floor(len(point_clouds) / num_clouds)).astype(np.int32)
    divis_indices = np.arange(0, len(point_clouds), divis_clouds)
    return batch_splitting(point_clouds, divis_indices, divis_clouds)

def splitting(array, start, step):
    array = jax.lax.dynamic_slice_in_dim(array, start, step)
    return array
batch_splitting = jax.vmap(splitting, in_axes=[None, 0, None])

def assign_primitive(shapes, point_cloud):
    sdf_min = opt_T(shapes[0], point_cloud)
    assigned_shape = shapes[0]
    for i in range(1, len(shapes)):
        sdf_current = opt_T(shapes[i], point_cloud)
        if sdf_current[1] < sdf_min[1]:
            sdf_min = sdf_current
            assigned_shape = shapes[i]
    return sdf_min[0], sdf_min[1], assigned_shape

def main(frames):
    test_shapes = ['square', 'circle']
    twist = get_robot_vel()
    measured_point_cloud = get_point_cloud(0)
    transform, sdf, shape = assign_primitive(test_shapes, measured_point_cloud)
    i = 1
    while i<frames:
        est_pos_SE2 = evolve_pos(transform, twist)
        est_pos = np.array([(est_pos_SE2).translation()[0],
                            (est_pos_SE2).translation()[1]])
        measured_point_cloud = get_point_cloud(i)

        if shape == 'circle' or shape == 'Circle' or shape == 'c' or shape == 'C':
            sdf_current = calc_cost_c(est_pos, measured_point_cloud)
        elif shape == 'square' or shape == 'Square'  or shape == 'sq' or shape == 'Sq':
            sdf_current = calc_cost_sq(est_pos, measured_point_cloud)
        else:
            return 'unrecognized shape'
        
        if sdf_current>.1:
            transform_new, sdf_new, shape_new = assign_primitive(test_shapes, measured_point_cloud)
            del_twist = get_twist(transform_new, transform)
            twist = twist + del_twist
            transform, sdf, shape = transform_new, sdf_new, shape_new
        else:
            transform, sdf, shape = est_pos, sdf_current, shape
        
        print("** iteration " + str(i) + " **")
        print('transform:', transform)
        print('sdf: ', sdf)
        
        i += 1

################################
## lie group helpers ###########
################################

def evolve_pos(R0, wt):
    R0 = SE2.from_xy_theta(R0[0], R0[1], 0.)
    Rf = R0 @ SE2.exp(wt)
    return Rf

def get_twist(R0, Rf):
    R0 = SE2.from_xy_theta(R0[0], R0[1], 0.)
    Rf = SE2.from_xy_theta(Rf[0], Rf[1], 0.)
    twist = SE2.log(Rf @ SE2.inverse(R0))
    return twist

################################
## test data ###################
################################

def get_point_cloud_circ(iter, start_pose, velocity):
    return get_circle(start_pose + np.array([iter*velocity, iter*velocity]), 1, 1000)
    
def get_point_cloud_rect(iter, start_pose, velocity):
    return get_rect(start_pose + np.array([iter*velocity, iter*velocity]), 1, 1, 1000)

batch_cloud_sq = jax.vmap(get_point_cloud_circ, in_axes=[None, 0, 0])
batch_cloud_c = jax.vmap(get_point_cloud_rect, in_axes=[None, 0, 0])
num_c = numpy.random.randint(0, global_num_clouds)

start_poses = (numpy.random.uniform(-10, 10, (global_num_clouds, 2))).reshape(5,2)
velocities = numpy.random.uniform(-.5, .5, global_num_clouds)

def get_point_clouds(iter):
    point_clouds_c = batch_cloud_c(iter, start_poses[0:num_c], velocities[0:num_c])
    point_clouds_c = np.ravel(point_clouds_c)
    point_clouds_c = np.reshape(point_clouds_c, (int(len(point_clouds_c)/2), 2))
    point_clouds_sq = batch_cloud_sq(iter, start_poses[num_c:global_num_clouds], velocities[num_c:global_num_clouds])
    point_clouds_sq = np.ravel(point_clouds_sq)
    point_clouds_sq= np.reshape(point_clouds_sq, (int(len(point_clouds_sq)/2), 2))

    return np.concatenate((point_clouds_c, point_clouds_sq), axis=0)

def get_robot_vel():
    Ti = np.array([0., 0., 0.])
    Tf = np.array([2., 2., 0.])
    return get_twist(Ti, Tf)

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

clouds = get_point_clouds(0)
start_time = time.time()
test_shape = 'square'
x = opt_T(test_shape, clouds)
print('done')

#TODO: implement T[2] into optimized T (theta isn't current used),