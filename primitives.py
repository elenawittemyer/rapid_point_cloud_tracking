import jax.numpy as np
from jax import jit, vmap
from sample_point_clouds import get_triangle, get_rhomb
import matplotlib.pyplot as plt

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

def sdfTriangle(p, l):
    r = l/np.cos(np.pi/6)
    q = r * np.array([.5, np.cos(np.pi/6)])
    q.at[1].set(-p[1]+q[1]*2/3)
    q.at[0].set(np.abs(q[0]))

    a = p-q*np.clip(np.dot(p,q)/np.dot(q,q), 0.0, 1.0)
    b = p-q*np.array([np.clip(p[0]/q[0], 0.0, 1.0), 1.0])
    s = -np.sign(q[1])
    d = np.minimum(np.array([np.dot(a, a), s*(p[0]*q[1]-p[1]*q[0])]), np.array([np.dot(b, b), s*(p[1]-q[1])]))
    return -np.sqrt(d[0])*np.sign(d[1])
batch_sdf_t = vmap(sdfTriangle, in_axes=[0, None])

def ndot(a, b):
    return a[0]*b[0]-a[1]*b[1]

def sdfRhombus(p, b):
    p = np.abs(p)
    h = np.clip(ndot(b-2*p, b)/np.dot(b, b), -1.0, 1.0)
    d = np.linalg.norm(p-.5*b*np.array([1-h, 1+h]))
    return d * np.sign(p[0]*b[1]+p[1]*b[0] - b[0]*b[1])
batch_sdf_r = vmap(sdfRhombus, in_axes=[0, None])

'''
batch_sdf_t_test = vmap(sdfTriangle, in_axes=[None, 0])
sample_triangle = get_triangle(np.array([0, 0]), 1, 100)

list_grid = []
p_x = np.arange(0, 10, .1)
p_y = np.arange(0, 10, .1)
coords = []
for j in range(len(p_y)):
    list_grid.append(np.hstack((np.array([p_x]).T, 
                               np.array([p_y[j]*np.ones(len(p_x))]).T)))
grid = list_grid[0]
for i in range(1, len(list_grid)):
    grid = np.vstack((grid, list_grid[i]))

sdf_array = []
sdf_sum = 0
for i in range(len(sample_triangle)):
    sdf_sum += batch_sdf_t_test(sample_triangle[i], grid)
sdf_grid = np.reshape(sdf_sum, (100, 100))
plt.imshow(sdf_grid)
plt.show()
'''