from math import *
import io, os, sys, time
import array as arr

def leg(n, p):
    ans, i = 0, p
    while i <= n:
        ans += n // i
        i *= p
    return ans

if __name__ == '__main__':
    for t in range(int(input())):
        n, p = map(int, input().split())
        print(leg(n, p))