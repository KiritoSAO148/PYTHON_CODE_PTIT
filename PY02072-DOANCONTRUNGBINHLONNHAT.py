from math import *
import io, os, sys, time
import array as arr

def max_avg_len(a):
    ans, cnt, res = 0, 0, max(a)
    for i in range(len(a)):
        if a[i] == res: cnt += 1
        else:
            ans = max(ans, cnt)
            cnt = 0
    ans = max(ans, cnt)
    cnt = 0
    return ans

if __name__ == '__main__':
    n = int(input())
    a = list(map(int, input().split()))
    print(max_avg_len(a))