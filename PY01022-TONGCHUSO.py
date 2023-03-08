from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    n = input()
    s, cnt = 0, 1
    for x in n: s += (ord(x) - 48)
    while s > 9:
        res = 0
        for x in str(s): res += (ord(x) - 48)
        s = res
        cnt += 1
    print(cnt)