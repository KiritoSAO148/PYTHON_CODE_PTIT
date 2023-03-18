from math import *
import io, os, sys, time
import array as arr
from sys import stdin

def solve(s, n):
    idx = -1
    for i in range(n - 2, -1, -1):
        if int(s[i]) > int(s[i + 1]):
            idx = i
            break
    res = -1
    for i in range(n - 1, idx, -1):
        if (res == -1 and int(s[i]) < int(s[idx])): res = i
        elif (idx > -1 and int(s[i]) >= int(s[res]) and int(s[i]) < int(s[idx])): res = i
    if idx == -1: return "".join("-1")
    else:
        (s[idx], s[res]) = (s[res], s[idx])
    return "".join(s)

if __name__ == '__main__':
    for _ in range(int(stdin.readline())):
        n = stdin.readline().strip('\n')
        ans = solve(list(n), len(n))
        if ans[0] == '0': print('-1')
        else: print(ans)