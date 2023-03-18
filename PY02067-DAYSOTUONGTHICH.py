from math import *
import io, os, sys, time
import array as arr

def check(a, pos):
    for i in range(len(a)):
        if a[i] // pos == a[i] // (pos + 1): return False
    return True

if __name__ == '__main__':
    n = int(input())
    a = list(map(int, input().split()))
    ans = 0
    b = a[::]
    for i in range(b[0], -1, -1):
        if check(a, i):
            for j in range(n): ans += a[j] // (i + 1) + 1
            break
    print(ans)