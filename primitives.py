import jax.numpy as np
from jax import jit, vmap
from sample_point_clouds import get_triangle

def sdfCircle(p, r): # p = point, r = radius
    return np.linalg.norm(p) - r

def sdfBox(p_array, a, b, t): # p = point, a = start of centerline, b = end of centerline, th = width
    sdf_array = []
    for i in range(len(p_array)):
        p = p_array[i]
        l = np.linalg.norm(b-a)
        d = (b-a)/l
        q = (p-(a+b)*.5)
        q = np.array([[d[0],-d[1]],
                    [d[1],d[0]]])@q
        q = np.abs(q)-np.array([l,t])*0.5
        sdf = np.linalg.norm(np.maximum(q, 0.)) + np.min(np.max(q[0],q[1]), 0.)
        sdf_array.append(sdf)
    return np.array(sdf_array)

def sdfTriangle(p, r):
    k = np.sqrt(3.0)
    p_x = abs(p[0]) - r
    p_y = p[1] + r/k

    pos_point = np.array([p_x-k*p_y,-k*p_x-p_y])/2.0
    neg_point = np.array([p_x, p_y])
    point_choice = [pos_point, neg_point]

    index_choice = np.argmax(np.array([p_x+k*p_y, 1E-10]))
    point = point_choice[index_choice]

    p_clamp_x = point[0] - np.clip(point[0], -2*r, 0)
    p_clamp_y = point[1]
    p_clamp = np.array([p_clamp_x, p_clamp_y])

    return -np.linalg.norm((p_clamp))*np.sign(p_clamp[1])
batch_sdf_t = vmap(sdfTriangle, in_axes=[0, None])

def sdfIsoTriangle(p, q):
    p_x = np.abs(p[0])
    p_y = p[1]
    p = np.array([p_x, p_y])
    a = p-q*np.clip(np.dot(p,q)/np.dot(q,q), 0, 1)
    b = p-q*np.array([np.clip(p[0]/q[0], 0, 1), 1])
    s = np.sign(q[1])
    d = np.minimum(np.array([np.dot(a, a), s*(p[0]*q[1]-p[1]*q[0])]),
                   np.array([np.dot(b, b), s*(p[1]-q[1])]))
    return -1*np.sqrt(d[0])*np.sign(d[1])

sample_triangle = get_triangle(np.array([0, 0]), 1, 100)
print(sdfIsoTriangle(sample_triangle[0], np.array([1, 1])))


#TODO: sdfTriangle is accurate but cannot be vmapped due to indexing. sdfIsoTriangle might not be accurate (figure out what q is; maybe it
# is equal and unequal side lengths?). Fix one or both of these.
