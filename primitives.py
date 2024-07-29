import jax.numpy as np
import numpy as onp
from jax import jit, vmap
from sample_point_clouds import get_hex, get_pent, get_triangle, get_rhomb, get_circle, get_rect
import matplotlib.pyplot as plt

#Works
def sdfCircle(p, r): # p = point, r = radius
    return np.linalg.norm(p) - r

#Works
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

def ndot(a, b):
    return a[0]*b[0]-a[1]*b[1]

#Works
def sdfRhombus(p, b):
    p = np.abs(p)
    h = np.clip(ndot(b-2*p, b)/np.dot(b, b), -1.0, 1.0)
    d = np.linalg.norm(p-.5*b*np.array([1-h, 1+h]))
    return d * np.sign(p[0]*b[1]+p[1]*b[0] - b[0]*b[1])
batch_sdf_r = vmap(sdfRhombus, in_axes=[0, None])

#Works
def sdfHexagon(p, l):
    r = l/(2*np.tan(np.pi/6))
    k = np.array([-0.866025404,0.5,0.577350269])
    p = np.abs(p)
    p = p - 2*np.min(np.array([np.dot(np.array([k[0], k[1]]), p), 0.0]))*np.array([k[0], k[1]])
    p = p - np.array([np.clip(p[0], -k[2]*r, k[2]*r), r])
    return np.linalg.norm(p) * np.sign(p[1])
batch_sdf_h = vmap(sdfHexagon, in_axes = [0, None])

#Works
def sdfTriangle(p, q):
    q = np.array([q[0]/2, -q[1]]) #for equilateral size 1, q=(.5, -1)
    p = np.array([np.abs(p[0]), p[1]-.5]) #p[1]-.5 only works for triangles with height 1
    a = p - q*np.clip(np.dot(p, q)/np.dot(q, q), 0, 1)
    b = p - q*np.array([np.clip(p[0]/q[0], 0, 1), 1])
    k = np.sign(q[1])
    d = np.min(np.array([np.dot(a, a), np.dot(b, b)]))
    s = np.max(np.array([k*(p[0]*q[1]-p[1]*q[0]), k*(p[1]-q[1])]))
    return np.sqrt(d)*np.sign(s)
batch_sdf_t = vmap(sdfTriangle, in_axes=[0, None])

#sample_triangle = get_triangle(np.array([0, 0]), 1, 1, 100)
#sdf_val = batch_sdf_t(sample_triangle, np.array([1, 1]))
'''
batch_sdf_t_test = vmap(sdfTriangle, in_axes=[0, None])
sample_triangle = get_triangle_2(np.array([0, 0]), 1, 1, 100)

list_grid = []
p_x = np.arange(-1, 1, .01)
p_y = np.arange(-1, 1, .01)
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
    sdf_sum += batch_sdf_t_test(grid, np.array([1, 1]))
sdf_grid = np.reshape(sdf_sum, (200, 200))
new_grid = onp.copy(sdf_grid)
min_idx = new_grid>-.15
max_idx = new_grid<.15
idx = np.where(np.logical_and(min_idx, max_idx))

new_grid[idx]+=100
plt.imshow(new_grid)
plt.show()
'''