import jax.numpy as np
from jax import jit
from functools import partial

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

a_test = np.array([0, 0])
b_test = np.array([0, 1])
th_test = 1