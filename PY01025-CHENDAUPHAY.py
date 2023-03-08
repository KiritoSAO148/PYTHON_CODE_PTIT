from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    n = input()
    ans, idx = '', 1
    for i in range(len(n) - 1, -1, -1):
        ans += n[i]
        if idx % 3 == 0: ans = ans + ','
        idx += 1
    a = []
    for i in range(len(ans) - 1, -1, -1):
        if i == len(ans) - 1 and ans[i] == ',': continue
        a.append(ans[i])
    for x in a: print(x, end = '')