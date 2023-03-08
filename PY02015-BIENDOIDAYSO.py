from math import *
import io, os, sys, time
import array as arr

def check(a):
    return a[0] == a[1] == a[2] == a[3]

if __name__ == '__main__':
    while True:
        a = list(map(int, sys.stdin.readline().split()))
        if a[0] == 0 and check(a): break
        cnt = 0
        while not check(a):
            tmp = a[0]
            a[0] = abs(a[0] - a[1])
            a[1] = abs(a[1] - a[2])
            a[2] = abs(a[2] - a[3])
            a[3] = abs(a[3] - tmp)
            cnt += 1
        print(cnt)