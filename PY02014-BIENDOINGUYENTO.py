from math import *
import io, os, sys, time
import array as arr

p = [1] * 10001
prime = []

def sieve():
    p[0] = p[1] = 0
    for i in range(2, isqrt(10000) + 1):
        if p[i]:
            for j in range(i * i, 10001, i): p[j] = 0
    for i in range(10001):
        if p[i]: prime.append(i)

if __name__ == '__main__':
    sieve()
    n = int(input())
    a = list(map(int, input().split()))
    ans = 0
    for x in a:
        sum = sys.maxsize
        for y in prime:
            sum = min(sum, abs(x - y))
        ans = max(ans, sum)
    print(ans)