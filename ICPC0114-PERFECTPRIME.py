from math import isqrt

def prime(n):
    if n < 2: return False
    for i in range(2, isqrt(n) + 1):
        if n % i == 0: return False
    return True

def check(n):
    s, m, tmp = 0, n, 0
    while m != 0:
        r = m % 10
        tmp = tmp * 10 + r
        if r != 2 and r != 3 and r != 5 and r != 7: return False
        s += r
        m //= 10
    if not prime(s): return False
    return prime(n) and prime(tmp)

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = int(input())
        if check(n): print('Yes')
        else: print('No')