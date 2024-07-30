import jax
import jaxopt
import jax.numpy as np
import numpy
from jax import jit
import jaxlie
from jaxlie import SE2, SE3
from sample_point_clouds import get_hex, get_rect, get_circle, get_rhomb, get_triangle
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

@jit
def sdf_t(T, point):
    T = SE2.from_xy_theta(T[0], T[1], 0.)
    T_inv = SE2.inverse(T)
    T_p = SE2.from_xy_theta(point[0], point[1], 0.)
    point_pos = np.array([(T_inv@T_p).translation()[0],
                            (T_inv@T_p).translation()[1]])
    
    p = point_pos
    q = np.array([1, -2])
    p = np.array([np.abs(p[0]), p[1]-1])
    a = p - q*np.clip(np.dot(p, q)/np.dot(q, q), 0, 1)
    b = p - q*np.array([np.clip(p[0]/q[0], 0, 1), 1])
    k = np.sign(q[1])
    d = np.min(np.array([np.dot(a, a), np.dot(b, b)]))
    s = np.max(np.array([k*(p[0]*q[1]-p[1]*q[0]), k*(p[1]-q[1])]))
    return np.sqrt(d)*np.sign(s)

def ndot(a, b):
    return a[0]*b[0]-a[1]*b[1]

@jit
def sdf_r(T, point):
    T = SE2.from_xy_theta(T[0], T[1], 0.)
    T_inv = SE2.inverse(T)
    T_p = SE2.from_xy_theta(point[0], point[1], 0.)
    point_pos = np.array([(T_inv@T_p).translation()[0],
                            (T_inv@T_p).translation()[1]])
    
    p = point_pos
    b = np.array([1, 1])
    p = np.abs(p)
    h = np.clip(ndot(b-2*p, b)/np.dot(b, b), -1.0, 1.0)
    d = np.linalg.norm(p-.5*b*np.array([1-h, 1+h]))
    return d * np.sign(p[0]*b[1]+p[1]*b[0] - b[0]*b[1])

@jit
def sdf_h(T, point):
    T = SE2.from_xy_theta(T[0], T[1], 0.)
    T_inv = SE2.inverse(T)
    T_p = SE2.from_xy_theta(point[0], point[1], 0.)
    point_pos = np.array([(T_inv@T_p).translation()[0],
                            (T_inv@T_p).translation()[1]])
    
    p = point_pos
    r = 1/(2*np.tan(np.pi/6))
    k = np.array([-0.866025404,0.5,0.577350269])
    p = np.abs(p)
    p = p - 2*np.min(np.array([np.dot(np.array([k[0], k[1]]), p), 0.0]))*np.array([k[0], k[1]])
    p = p - np.array([np.clip(p[0], -k[2]*r, k[2]*r), r])
    return np.linalg.norm(p) * np.sign(p[1])

batch_sdf_sq = jax.vmap(sdf_sq, in_axes=[None, 0])
batch_sdf_c = jax.vmap(sdf_c, in_axes=[None, 0])
batch_sdf_t = jax.vmap(sdf_t, in_axes=[None, 0])
batch_sdf_r = jax.vmap(sdf_r, in_axes=[None, 0])
batch_sdf_h = jax.vmap(sdf_h, in_axes=[None, 0])

def calc_cost_c(T, point_cloud):
    sdf_array = batch_sdf_c(T, point_cloud)
    cost = np.linalg.norm(np.array(sdf_array), ord=1)
    return cost

def calc_cost_sq(T, point_cloud):
    sdf_array = batch_sdf_sq(T, point_cloud)
    cost = np.linalg.norm(np.array(sdf_array), ord=1)
    return cost

def calc_cost_t(T, point_cloud):
    sdf_array = batch_sdf_t(T, point_cloud)
    cost = np.linalg.norm(np.array(sdf_array), ord=1)
    return cost

def calc_cost_r(T, point_cloud):
    sdf_array = batch_sdf_r(T, point_cloud)
    cost = np.linalg.norm(np.array(sdf_array), ord=1)
    return cost

def calc_cost_h(T, point_cloud):
    sdf_array = batch_sdf_h(T, point_cloud)
    cost = np.linalg.norm(np.array(sdf_array), ord=1)
    return cost

def opt_T(shape, point_cloud):
    if shape == 'circle' or shape == 'Circle' or shape == 'c' or shape == 'C':
        solver = jaxopt.ScipyMinimize(method = 'Nelder-Mead', fun = calc_cost_c, maxiter = 500)
        T_opt, state = solver.run(np.array([0., 0.]), point_cloud)
    elif shape == 'square' or shape == 'Square'  or shape == 'sq' or shape == 'Sq':
        solver = jaxopt.ScipyMinimize(method = 'Nelder-Mead', fun = calc_cost_sq, maxiter = 500)
        T_opt, state = solver.run(np.array([0., 0.]), point_cloud)
    elif shape == 'triangle' or shape == 'Triangle' or shape == 't' or shape == 'T':
        solver = jaxopt.ScipyMinimize(method = 'Nelder-Mead', fun = calc_cost_t, maxiter = 500)
        T_opt, state = solver.run(np.array([0., 0.]), point_cloud)
    elif shape == 'rhombus' or shape == 'Rhombus' or shape =='r' or shape == 'R':
        solver = jaxopt.ScipyMinimize(method = 'Nelder-Mead', fun = calc_cost_r, maxiter = 500)
        T_opt, state = solver.run(np.array([0., 0.]), point_cloud)
    elif shape == 'hexagon' or shape=='Hexagon' or shape == 'h' or shape == 'H':
        solver = jaxopt.ScipyMinimize(method = 'Nelder-Mead', fun = calc_cost_h, maxiter = 500)
        T_opt, state = solver.run(np.array([0., 0.]), point_cloud)
    else:
        return 'unrecognized shape'
    
    return T_opt, state.fun_val

def assign_primitive(shapes, point_cloud):
    min_params = opt_T(shapes[0], point_cloud)
    assigned_shape = shapes[0]
    for i in range(1, len(shapes)):
        current_params = opt_T(shapes[i], point_cloud)
        if current_params[1] < min_params[1]:
            min_params = current_params
            assigned_shape = shapes[i]
    return min_params[0], min_params[1], assigned_shape

def main(frames, num_clouds):
    test_shapes = ['square', 'circle', 'triangle', 'rhombus', 'hexagon']
    transforms = []
    sdfs = []
    shapes = []
    twists = get_robot_vel()*np.ones((num_clouds, 3))
    measured_clouds, pos_data, vel_data = get_point_clouds(0)
    segmented_clouds, init_split_idx = sdf_segmentation(np.array([0., 0.]), measured_clouds, num_clouds)
    
    for cloud in segmented_clouds:
        transform, sdf, shape = assign_primitive(test_shapes, cloud)
        transforms.append(transform)
        sdfs.append(sdf)
        shapes.append(shape)
    init_shapes = shapes

    i = 1
    while i<frames:
        measured_clouds, pos_data, vel_data = get_point_clouds(i, pos_data, vel_data)
        segmented_clouds, ordered_shapes = shape_segmentation(init_shapes, num_clouds, init_split_idx, measured_clouds)
        
        transforms_new = []
        sdfs_new = []

        for j in range(num_clouds):
            transform = transforms[j]
            est_pos_SE2 = evolve_pos(transform, twists[j])
            est_pos = np.array([(est_pos_SE2).translation()[0],
                                (est_pos_SE2).translation()[1]])

            if ordered_shapes[j] == 'circle':
                sdf_current = calc_cost_c(est_pos, segmented_clouds[j])
                
            elif ordered_shapes[j] == 'square':
                sdf_current = calc_cost_sq(est_pos, segmented_clouds[j])

            elif ordered_shapes[j] == 'triangle':
                sdf_current = calc_cost_t(est_pos, segmented_clouds[j])
            
            elif ordered_shapes[j] == 'rhombus':
                sdf_current = calc_cost_r(est_pos, segmented_clouds[j])

            elif ordered_shapes[j] == 'hexagon':
                sdf_current = calc_cost_h(est_pos, segmented_clouds[j])

            else:
                return 'unrecognized shape'
            
            if sdf_current>.1:
                transform_new, sdf_new, shape_new = assign_primitive([ordered_shapes[j]], segmented_clouds[j])
                del_twist = get_twist(transform, transform_new)
                twists = twists.at[j].set(get_robot_vel() + del_twist)
                transform, sdf, shape = transform_new, sdf_new, shape_new
            else:
                transform, sdf, shape = est_pos, sdf_current, shape

            transforms_new.append(transform)
            sdfs_new.append(sdf)
            
        print("\n** iteration " + str(i) + " **")
        print('transforms: ', np.array([transforms])[0])
        print('sdfs: ', np.array([sdfs])[0])
        print('twists :', twists)

        transforms = transforms_new
        sdfs = sdfs_new

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
    split_indices = np.array(split_indices)+1
    split_clouds = [point_cloud[0:split_indices[0]]]
    for i in range(0, num_clouds-1):
        split_clouds.append(point_cloud[(split_indices[i]):(split_indices[i+1]-1)])
    return split_clouds, split_indices

def shape_segmentation(shapes, num_clouds, split_indices, point_cloud): #NOTE: only works if point clouds remain same size and are measured sequentially
    split_clouds = []
    split_clouds = [point_cloud[0:split_indices[0]]]
    for i in range(0, num_clouds-1):
        split_clouds.append(point_cloud[(split_indices[i]):(split_indices[i+1]-1)])
    
    swap_idxs = []
    new_shapes = []
    for j in range(num_clouds):
        T, val = opt_T(shapes[0], split_clouds[j])
        sdf_options = []
        shape_options = []
        if 'circle' in shapes:
            sdf_options.append(calc_cost_c(T, split_clouds[j]))
            shape_options.append('circle')
        if 'square' in shapes:
            sdf_options.append(calc_cost_sq(T, split_clouds[j]))
            shape_options.append('square')
        if 'triangle' in shapes:
            sdf_options.append(calc_cost_t(T, split_clouds[j]))
            shape_options.append('triangle')
        if 'rhombus' in shapes:
            sdf_options.append(calc_cost_r(T, split_clouds[j]))
            shape_options.append('rhombus')
        if 'hexagon' in shapes:
            sdf_options.append(calc_cost_h(T, split_clouds[j]))
            shape_options.append('hexagon')
        
        sdf_options = np.array(sdf_options)
        min_sdf = np.min(sdf_options)
        min_idx = np.where(sdf_options == min_sdf)[0][0]
        swap_idxs.append(min_idx)
        new_shapes.append(shape_options[min_idx])
    
    new_split_clouds = []
    for i in range(len(swap_idxs)):
        new_split_clouds.append(split_clouds[swap_idxs[i]])

    return new_split_clouds, new_shapes


################################
## test data ###################
################################
def get_point_clouds(iter, start_pos = None, vel = None):
    num_clouds = 5
    if start_pos is None:
        start_pos_x = numpy.random.choice(np.arange(-10, 11), num_clouds, replace=False)
        start_pos_y = numpy.random.choice(np.arange(-10, 11), num_clouds, replace=False)
        start_pos = np.hstack((np.array([start_pos_x]).T, np.array([start_pos_y]).T))

    if iter%10 == 0:
        vel = numpy.random.randint(-2, 3, num_clouds*2)
        vel = np.reshape(vel, (num_clouds, 2))
    
    start_pos = start_pos + vel
    point_cloud_c = get_circle(start_pos[0], 1, 100)
    point_cloud_sq = get_rect(start_pos[1], 1, 1, 100)
    point_cloud_t = get_triangle(start_pos[2], 2, 2, 100)
    point_cloud_r = get_rhomb(start_pos[3], 1, 1, 100)
    point_cloud_h = get_hex(start_pos[4], 1, 100)
    return np.concatenate((point_cloud_c, point_cloud_sq, point_cloud_t, point_cloud_r, point_cloud_h), axis=0), start_pos, vel

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
main(20, 5)
print(time.time()-start_time)
