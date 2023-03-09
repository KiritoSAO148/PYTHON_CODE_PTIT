from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    n = int(input())
    a = list(map(int, input().split()))
    sum, ans = sys.maxsize, 0
    for x in a:
        res = 0
        for y in a:
            res += abs(x - y)
        if res < sum:
            sum = res
            ans = x
    print(sum, ans, end = ' ')