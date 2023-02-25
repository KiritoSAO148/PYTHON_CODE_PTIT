from math import *

def prime(n):
    if n < 2: return False
    for i in range(2, isqrt(n) + 1):
        if n % i == 0: return False
    return True

def check(n):
    if not prime(len(n)): return False
    c1, c2 = 0, 0
    for x in n:
        if prime(ord(x) - 48): c1 += 1
        else: c2 += 1
    return c1 > c2

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        if check(n): print('YES')
        else: print('NO')