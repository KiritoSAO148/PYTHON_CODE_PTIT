from math import *

p = [1] * 1000001

def sieve():
    p[0] = p[1] = 0
    for i in range (2, isqrt(1000000) + 1):
        if p[i]:
            for j in range (i * i, 1000000, i):
                p[j] = 0

def sum(n):
    ans = 0
    while n > 0:
        ans += n % 10
        n //= 10
    return ans

if __name__ == '__main__':
    TC = int(input())
    sieve()
    for t in range(TC):
        a, b = map(int, input().split())
        if p[sum(gcd(a, b))]: print('YES')
        else: print('NO')