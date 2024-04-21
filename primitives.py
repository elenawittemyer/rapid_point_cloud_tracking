import jax.numpy as np

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

    pos_point = np.array([p[0]-k*p[1],-k*p[0]-p[1]])/2.0
    neg_point = np.array([p_x, p_y])
    point_choice = [pos_point, neg_point]

    index_choice = np.argmax(np.array([p_x+k*p_y, 0.0]))
    point = point_choice[index_choice]

    p_clamp_x = np.min(np.array([np.max(np.array([point[0], -2.0*r])), 0.0]))
    p_clamp_y = point[1]
    p_clamp = np.array([p_clamp_x, p_clamp_y])

    return -np.linalg.norm((p_clamp))*np.sign(p_clamp[1])

a_test = np.array([0, 0])
b_test = np.array([0, 1])
th_test = 1

p_test = [2,2]
r_test = 1
