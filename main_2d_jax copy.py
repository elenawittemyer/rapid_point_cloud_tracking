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

def main(frames, num_clouds):
    test_shapes = ['square', 'circle']
    transforms = []
    sdfs = []
    shapes = []
    twist = get_robot_vel()
    measured_clouds = get_point_clouds(0)
    segmented_clouds = sdf_segmentation(np.array([0., 0.]), measured_clouds, num_clouds)
    
    for cloud in segmented_clouds:
        transform, sdf, shape = assign_primitive(test_shapes, cloud)
        transforms.append(transform)
        sdfs.append(sdf)
        shapes.append(shape)

    i = 1
    while i<frames:
        measured_clouds = get_point_clouds(i)
        segmented_clouds = sdf_segmentation(np.array([0., 0.]), measured_clouds, num_clouds)
        for j in range(num_clouds):
            cloud_options = range(num_clouds)

            est_pos_SE2 = evolve_pos(transforms[j], twist)
            est_pos = np.array([(est_pos_SE2).translation()[0],
                                (est_pos_SE2).translation()[1]])

            if shape == 'circle' or shape == 'Circle' or shape == 'c' or shape == 'C':
                sdf_options = []
                for k in cloud_options:
                    sdf_options.append(calc_cost_c(est_pos, segmented_clouds[k]))
                sdf_options = np.array(sdf_options)
                sdf_current = np.min(sdf_options)
                min_index = np.where(sdf_options == sdf_current)
                cloud_options = cloud_options.tolist()
                cloud_options.pop(min_index)
                cloud_options = np.array(cloud_options)
                
            elif shape == 'square' or shape == 'Square'  or shape == 'sq' or shape == 'Sq':
                sdf_options = []
                for k in cloud_options:
                    sdf_options.append(calc_cost_sq(est_pos, segmented_clouds[k]))
                sdf_options = np.array(sdf_options)
                sdf_current = np.min(sdf_options)
                min_index = np.where(sdf_options == sdf_current)
                cloud_options = cloud_options.tolist()
                cloud_options.pop(min_index)
                cloud_options = np.array(cloud_options)
            else:
                return 'unrecognized shape'
        
            if sdf_current>.1:
                transform_new, sdf_new, shape_new = assign_primitive(test_shapes, measured_clouds[min_index])
                del_twist = get_twist(transform_new, transform)
                twist = twist + del_twist
                transform, sdf, shape = transform_new, sdf_new, shape_new
            else:
                transform, sdf, shape = est_pos, sdf_current, shape

            transforms.append(transform)
            sdfs.append(sdf)
            shapes.append(shape)
            
        print("** iteration " + str(i) + " **")
        print('transforms:', transforms)
        print('sdfs: ', sdfs)
            
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
## segmentation ################
################################

def sdf_segmentation(transform_init, point_cloud, num_clouds):
    sdf_array_c = batch_sdf_c(transform_init, point_cloud)
    filter = np.diff(sdf_array_c)
    split_indices = np.where(np.abs(filter)>2.5)[0].tolist()
    split_indices.append(len(point_cloud))
    split_indices = np.array(split_indices)
    split_clouds = [point_cloud[0:split_indices[0]]]
    for i in range(0, num_clouds-1):
        split_clouds.append(point_cloud[(split_indices[i]):(split_indices[i+1])])
    return split_clouds

################################
## test data ###################
################################

def get_point_clouds(iter):
    point_cloud_c1 = get_circle(np.array([1., 1.]), 1, 50)
    point_cloud_sq1 = get_rect(np.array([9., 5.]), 1, 1, 50)
    return np.concatenate((point_cloud_c1, point_cloud_sq1), axis=0)

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

#TODO: implement T[2] into optimized T (theta isn't current used)

start_time = time.time()
main(2, 2)
print(time.time()-start_time)