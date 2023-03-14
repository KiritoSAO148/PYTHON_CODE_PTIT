from math import *
import io, os, sys, time
import array as arr

def solve(n):
    cnt, l = 0, 1
    while l * (l + 1) < n * 2:
        res = (1.0 * n - (l * (l + 1)) / 2) / (l + 1)
        if res - int(res) == 0: cnt += 1
        l += 1
    return cnt

if __name__ == '__main__':
    for _ in range(int(sys.stdin.readline())): print(solve(int(sys.stdin.readline())))