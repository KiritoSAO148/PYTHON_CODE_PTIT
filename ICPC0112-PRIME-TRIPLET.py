from math import *
import io, os, sys, time
import array as arr

p = [1] * 1000001

def sieve():
    p[0] = p[1] = 0
    for i in range(2, isqrt(1000000) + 1):
        if p[i]:
            for j in range(i * i, 1000001, i): p[j] = 0

if __name__ == '__main__':
    sieve()
    for t in range(int(input())):
        n = int(input())
        cnt = 0
        for i in range(2, n):
            if (i + 2 < n and i + 6 < n and p[i] and p[i + 2] and p[i + 6]) or (i + 4 < n and i + 6 < n and p[i] and p[i + 4] and p[i + 6]): cnt += 1
        print(cnt)