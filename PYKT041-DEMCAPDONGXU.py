from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    n = int(input())
    a = [''] * n
    for i in range(n): a[i] = input()
    cnt1, cnt2 = [0] * n, [0] * n
    for i in range(n):
        for j in range(n):
            if a[i][j] == 'C':
                cnt1[i] += 1
                cnt2[j] += 1
    cnt = 0
    for i in range(n): cnt += (cnt1[i] * (cnt1[i] - 1) // 2) + (cnt2[i] * (cnt2[i] - 1) // 2)
    print(cnt)