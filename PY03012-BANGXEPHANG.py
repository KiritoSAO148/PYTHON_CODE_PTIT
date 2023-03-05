from math import *
import io, os, sys, time
import array as arr
from functools import cmp_to_key

def cmp(a, b):
    if a[1] != b[1]: return b[1] - a[1]
    if a[1] == b[1]: return a[2] - b[2]
    return a[0] - b[0]

if __name__ == '__main__':
    n = int(input())
    lst = []
    for i in range(n):
        res = []
        res.append(input())
        a, b = map(int, input().split())
        res.append(a)
        res.append(b)
        lst.append(res)
    lst.sort(key = cmp_to_key(cmp))
    for x in lst:
        for y in x: print(y, end = ' ')
        print()