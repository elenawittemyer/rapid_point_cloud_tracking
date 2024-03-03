import numpy as np
import point_cloud_utils as pcu

# 1000 random query points to compute the SDF at
query_pts = np.random.rand(1000, 3)

v, f = pcu.load_mesh_vf("cube.obj")

# sdf is the signed distance for each query point
# fid is the nearest face to each query point on the mesh
# bc are the barycentric coordinates of the nearest point to each query point within the face
sdf, fid, bc = pcu.signed_distance_to_mesh(query_pts, v, f)