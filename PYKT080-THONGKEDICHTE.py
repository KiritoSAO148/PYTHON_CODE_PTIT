from math import *
import io, os, sys, time
import array as arr
from sys import stdin, stdout

dx = [-1, -1, -1, 0, 1, 1, 1, 0]
dy = [-1, 0, 1, 1, 1, 0, -1, -1]

if __name__ == '__main__':
    n, m = map(int, stdin.readline().split())
    a = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    visited = [[False for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        res = list(map(int, stdin.readline().split()))
        for j in range(1, m + 1): a[i][j] = res[j - 1]
    sum = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i][j] == -1:
                for k in range(8):
                    i1, j1 = i + dx[k], j + dy[k]
                    if i1 >= 1 and i1 <= n and j1 >= 1 and j1 <= m and not visited[i1][j1]:
                        visited[i1][j1] = True
                        sum += a[i1][j1]
    print(sum)