from math import *
import io, os, sys, time
import array as arr
from sys import stdin, stdout

if __name__ == '__main__':
    tmp = []
    while True:
        try: tmp += input().split()
        except EOFError: break
    TC = int(tmp[0])
    idx = 1
    for _ in range(TC):
        a = []
        n, k = int(tmp[idx]), int(tmp[idx + 1])
        idx += 2
        for i in range(idx, idx + n): a.append(int(tmp[i]))
        idx += n
        res = n + 1
        for i in range(n):
            g = a[i]
            for j in range(i, n):
                g = gcd(g, a[j])
                if g == k:
                    res = min(res, j - i + 1)
                    break
                if g < k: break
        if res == n + 1:
            print(-1)
        else:
            print(res)