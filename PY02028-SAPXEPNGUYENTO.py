from math import *

p = [True] * 1001

def sieve():
    p[0] = p[1] = False
    for i in range(2, isqrt(1000) + 1):
        if p[i]:
            for j in range(i * i, 1001, i): p[j] = False

if __name__ == '__main__':
    sieve()
    n = int(input())
    a = list(map(int, input().split()))
    prime = [] * n
    for i in range(n):
        if p[a[i]]: prime.append(a[i])
    prime.sort(reverse = True)
    for i in range(n):
        if p[a[i]]:
            print(prime[-1], end = ' ')
            prime.pop()
        else: print(a[i], end = ' ')