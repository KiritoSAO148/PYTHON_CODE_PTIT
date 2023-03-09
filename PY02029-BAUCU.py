from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    d = {}
    for x in a:
        if x not in d: d[x] = 1
        else: d[x] += 1
    max1, max2, check, ans = 0, 0, 0, 0
    for x in d.keys(): max1 = max(max1, d[x])
    for key, val in d.items():
        if val > max2 and val < max1:
            max2 = val
            ans = key
            check = 1
    if not check: print('NONE')
    else: print(ans)