from math import *

def prime(n):
    if n < 2: return False
    for i in range(2, isqrt(n) + 1):
        if n % i == 0: return False
    return True

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = int(input())
        if prime(int(str(n)[0:3])) and prime(int(str(n)[-3:])): print('YES')
        else: print('NO')