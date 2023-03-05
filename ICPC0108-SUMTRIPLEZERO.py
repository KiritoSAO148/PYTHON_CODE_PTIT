from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    for t in range(int(input())):
        n = int(input())
        a = list(map(int, input().split()))
        cnt = 0
        a.sort()
        for i in range(n - 1):
            l, r = i + 1, n - 1
            while l < r:
                if a[i] + a[l] + a[r] == 0:
                    cnt += 1
                    l += 1
                elif a[i] + a[l] + a[r] < 0: l += 1
                else: r -= 1
        print(cnt)