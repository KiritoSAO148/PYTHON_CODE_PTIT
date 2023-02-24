from math import *

def prime(n):
    if n < 2: return False
    for i in range(2, isqrt(n) + 1):
        if n % i == 0: return False
    return True

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        s = 0
        for x in n: s += ord(x) - 48
        if prime(s): print('YES')
        else: print('NO')