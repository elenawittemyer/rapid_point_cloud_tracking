import jax
import jaxopt
import jax.numpy as np
import numpy
from jax import jit
import jaxlie
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
        solver = jaxopt.ScipyMinimize(method = 'Nelder-Mead', fun = calc_cost_c, maxiter = 500)
        T_opt, state = solver.run(np.array([0., 0.]), point_cloud)
    elif shape == 'square' or shape == 'Square'  or shape == 'sq' or shape == 'Sq':
        solver = jaxopt.ScipyMinimize(method = 'Nelder-Mead', fun = calc_cost_sq, maxiter = 500)
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

def get_point_cloud(iter):
    return get_circle(random_vel(iter), 1, 100)
    #return get_rect(random_vel(iter), 1, 1, 100)


def random_vel(iter):
    if iter<5:
        pos = np.array([iter+1, iter+1])
    elif 5<=iter<20:
        pos = np.array([2*iter-5, 11-iter])
    elif 20<=iter<28:
        pos = np.array([73-2*iter, iter-27])
    else:
        pos = np.array([iter-9, iter-29])
    return pos

def get_robot_vel():
    Ti = np.array([0., 0., 0.])
    Tf = np.array([0., 0., 0.])
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

#TODO: implement T[2] into optimized T (theta isn't current used)

start_time = time.time()
main(20)
print(time.time()-start_time)