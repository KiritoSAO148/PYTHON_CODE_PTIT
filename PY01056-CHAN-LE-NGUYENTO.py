from math import *

def prime(n):
    if n < 2: return False
    for i in range(2, isqrt(n) + 1):
        if n % i == 0: return False
    return True

def check(n):
    s = 0
    for i in range(0, len(n)):
        if i % 2 == 0:
            if (ord(n[i]) - 48) % 2 == 1: return False
        elif i % 2 == 1:
            if (ord(n[i]) - 48) % 2 == 0: return False
        s += ord(n[i]) - 48
    return prime(s)

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        if check(n): print('YES')
        else: print('NO')