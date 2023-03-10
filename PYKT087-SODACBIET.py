from math import *
import io, os, sys, time
import array as arr

MOD = 10 ** 9 + 7

def solve(n, k):
    ans, p = 0, 1
    while k > 0:
        if k % 2 == 1:
            ans += p
            ans %= MOD
        p *= n
        k //= 2
    return ans

if __name__ == '__main__':
    for _ in range(int(sys.stdin.readline())):
        n, k = map(int, sys.stdin.readline().split())
        print(solve(n, k))