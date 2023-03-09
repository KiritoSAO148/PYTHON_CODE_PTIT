from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    for _ in range(int(sys.stdin.readline())):
        n, m, k = map(int, sys.stdin.readline().split())
        a = list(map(int, sys.stdin.readline().split()))
        b = list(map(int, sys.stdin.readline().split()))
        c = list(map(int, sys.stdin.readline().split()))
        res, i, j, l = [], 0, 0, 0
        while i < n and j < m and l < k:
            if a[i] == b[j] and b[j] == c[l]:
                res.append(a[i])
                i += 1; j += 1; l += 1
            elif a[i] <= b[j] and a[i] <= c[l]: i += 1
            elif b[j] <= a[i] and b[j] <= c[l]: j += 1
            else: l += 1
        if len(res) == 0: print('NO')
        else:
            for x in res: print(x, end = ' ')
            print()