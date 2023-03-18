from math import *
import io, os, sys, time
import array as arr
from sys import stdin

if __name__ == '__main__':
    for _ in range(int(stdin.readline())):
        n, m = map(int, stdin.readline().split())
        a = [[0 for _ in range(m)] for _ in range(n)]
        b = [[0 for _ in range(3)] for _ in range(3)]
        for i in range(n): a[i] = list(map(int, stdin.readline().split()))
        for i in range(3): b[i] = list(map(int, stdin.readline().split()))
        res = [[0 for _ in range(m - 2)] for _ in range(n - 2)]
        for i in range(n - 2):
            for j in range(m - 2):
                s = 0
                for k in range(3):
                    for l in range(3):
                        s += a[i + k][j + l] * b[k][l]
                res[i].append(s)
        print(sum(sum(x) for x in res))