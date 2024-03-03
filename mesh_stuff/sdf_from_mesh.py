from open3d import core as o3c
import numpy as np
from sample_point_clouds import get_cube
from scipy.spatial.distance import cdist
import point_cloud_utils as pcu
import time

def sdf_from_point_cloud(point_cloud, robot_pos):
    summed_cloud = np.sum(point_cloud, axis=1)
    top_corner_val = np.max(summed_cloud)
    bottom_corner_val = np.min(summed_cloud)
    top_corner = np.where(summed_cloud==top_corner_val)[0][0]
    bottom_corner = np.where(summed_cloud==bottom_corner_val)[0][0]
    extent = (1/np.sqrt(3))*np.linalg.norm(point_cloud[top_corner]-point_cloud[bottom_corner])
    center = (point_cloud[top_corner] + point_cloud[bottom_corner])/2
    
    v, f = pcu.load_mesh_vf("cube.obj")
    v = extent*v + center
    sdf, fid, bc = pcu.signed_distance_to_mesh(robot_pos, v, f)
    return sdf, fid, bc

start_time = time.time()
cube_cloud = get_cube(np.array([5,5,5]), 10, .9)
pos = np.array([[0.,0.,0.]])
sdf = sdf_from_point_cloud(cube_cloud, pos)

print(sdf)




