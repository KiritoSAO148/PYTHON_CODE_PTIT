from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    n = int(input())
    a = []
    for i in range(n): a.append(input())
    res, tmp = [], []
    for i in range(n):
        if len(a[i].split()): tmp.append(a[i])
        else:
            res.append(tmp)
            tmp = []
    if len(tmp) > 0: res.append(tmp)
    for x in res: print(x[0], len(x) - 1, sep = ': ')