from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    n = int(sys.stdin.readline())
    b = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n): b[i] = list(map(int, sys.stdin.readline().split()))
    a, s = [], 0
    for i in range(n):
        s += sum(b[i])
        a.append(sum(b[i]))
    if n == 2:
        for x in a: print(x // 2, end = ' ')
    else:
        s = (s // 2 // (n - 1))
        for x in a: print(((x - s) // (n - 2)), end = ' ')