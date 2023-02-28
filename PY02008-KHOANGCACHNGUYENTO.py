from math import *

p = [True] * 10001
a = []

def sieve():
    p[0] = p[1] = False
    for i in range(2, isqrt(10000) + 1):
        if p[i]:
            for j in range(i * i, 10001, i): p[j] = False
    for i in range(10001):
        if p[i]: a.append(i)

if __name__ == '__main__':
    sieve()
    n, x = map(int, input().split())
    m = 0
    for i in range(n + 1):
        if i == 0: print(x, end = ' ')
        else:
            x += a[m]
            m += 1
            print(x, end = ' ')