from math import *

def prime (n):
    if n < 2: return False
    for i in range (2, isqrt(n) + 1):
        if n % i == 0: return False
    return True

t = int(input())
while t > 0:
    n = int(input())
    k = 0
    for x in range (1, n):
        if gcd(x, n) == 1: k += 1
    if prime(k): print('YES')
    else: print('NO')
    t -= 1