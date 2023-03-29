from math import *
import io, os, sys, time
import array as arr
from scipy.spatial import ConvexHull
import scipy.spatial.qhull
import numpy as np

if __name__ == '__main__':
    a = []
    while True:
        try:
            a.append(input().split())
        except EOFError: break
    b = []
    for x in a:
        for y in x: b.append(int(y))
    idx = 1
    del a
    for _ in range(b[0]):
        n = b[idx]
        idx += 1
        a = []
        for i in range(n):
            a.append((b[idx], b[idx + 1]))
            idx += 2
        a = np.array(a)
        hull = ConvexHull(a)
        points = a[hull.vertices]
        l = len(points)
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        area = 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(l - 1)) + x[l - 1] * y[0] - x[0] * y[l - 1])
        print('{:.4f}'.format(area))