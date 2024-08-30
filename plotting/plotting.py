import numpy as np
import matplotlib.pyplot as plt
import os
import json

plt.rc('font',family='Times New Roman')

def plot_den():
    d_c = []
    t_c = []
    f = open("plotting/solve_time_circle.txt")
    for l in f.readlines():
        val = l.split(" ")
        d_c.append(int(val[0]))
        t_c.append(float(val[1].split("\n")[0]))
    f.close()

    d_s = []
    t_s = []
    f = open("plotting/solve_time_square.txt")
    for l in f.readlines():
        val = l.split(" ")
        d_s.append(int(val[0]))
        t_s.append(float(val[1].split("\n")[0]))
    f.close()

    d_t = []
    t_t = []
    f = open("plotting/solve_time_triangle.txt")
    for l in f.readlines():
        val = l.split(" ")
        d_t.append(int(val[0]))
        t_t.append(float(val[1].split("\n")[0]))
    f.close()

    d_r = []
    t_r = []
    f = open("plotting/solve_time_rhombus.txt")
    for l in f.readlines():
        val = l.split(" ")
        d_r.append(int(val[0]))
        t_r.append(float(val[1].split("\n")[0]))
    f.close()

    d_h = []
    t_h = []
    f = open("plotting/solve_time_hexagon.txt")
    for l in f.readlines():
        val = l.split(" ")
        d_h.append(int(val[0]))
        t_h.append(float(val[1].split("\n")[0]))
    f.close()

    plt.plot(d_c, t_c)
    plt.plot(d_s, t_s)
    plt.plot(d_t, t_t)
    plt.plot(d_r, t_r)
    plt.plot(d_h, t_h)

    plt.ylim([.2, .45])
    font_title = {'fontname':'Times New Roman', 'size': 16}
    font_ax = {'fontname':'Times New Roman', 'size': 14}

    plt.legend(['Circle', 'Square', 'Triangle', 'Rhombus', 'Hexagon'])
    plt.xlabel('Density (points)', **font_ax)
    plt.ylabel('Average Solve Time (s)', **font_ax)
    plt.title('Point Cloud Density vs Solve Time', **font_title)

    plt.show()

    return 0

def plot_time_comp():
    num = []
    time_lg = []
    time_nlg = []
    f = open("plotting/solve_time_comparison.txt")
    for l in f.readlines():
        val = l.split(" ")
        num.append(int(val[0]))
        time_lg.append(float(val[1]))
        time_nlg.append(float(val[2].split("\n")[0]))

    plt.plot(num, time_lg)
    plt.plot(num, time_nlg)

    plt.xticks(range(1, 6))
    font_title = {'fontname':'Times New Roman', 'size': 16}
    font_ax = {'fontname':'Times New Roman', 'size': 14}

    plt.legend(['With Lie Groups', 'Without Lie Groups'])
    plt.xlabel('Number of Objects', **font_ax)
    plt.ylabel('Average Solve Time (s)', **font_ax)

    plt.title('Number of Objects vs Solve Time', **font_title)

    plt.show()
    return 0

def plot_accuracy():
    num_clouds = []
    acc_range = []
    f = open("plotting/accuracy.txt")
    for l in f.readlines():
        val = l.split(" ")
        num_clouds.append(int(val[0]))
        acc_range.append(np.array(json.loads(val[1].split("\n")[0])))
    
    labels = ['1', '2', '3', '4', '5']
    fig, ax = plt.subplots()
    bplot = ax.boxplot([acc_range[0], acc_range[1], acc_range[2], acc_range[3], acc_range[4]], notch=False, patch_artist=True)

    for patch in bplot['boxes']:
        patch.set_facecolor('teal')
    for median in bplot['medians']:
        median.set_color('black')

    font_title = {'fontname':'Times New Roman', 'size': 16}
    font_ax = {'fontname':'Times New Roman', 'size': 14}

    ax.set_xticklabels(labels)
    ax.set_xlabel('Number of Point Clouds', **font_ax)
    ax.set_ylabel('Reduction in SDF (%)', **font_ax)
    ax.set_title('Number of Objects vs SDF Reduction', **font_title)
    plt.show()

    return 0

plot_den()