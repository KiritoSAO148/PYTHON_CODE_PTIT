from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    n = int(input())
    a = [] * n
    for i in range(n): a.append(input())
    res = []
    i = 0
    while i < n:
        if i < n and len(a[i].split()) == 6:
            res.append(1)
            while i < n and len(a[i].split()) == 6:
                i += 2
        if i < n and len(a[i].split()) == 7:
            res.append(2)
            i += 4
    print(len(res))
    for x in res: print(x)