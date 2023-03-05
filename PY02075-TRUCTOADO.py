from math import *
import io, os, sys, time

if __name__ == '__main__':
    for t in range(int(input())):
        n = int(input())
        x = []
        for i in range(n): x.append(list(map(int, input().split())))
        x.sort(key = lambda x : x[1])
        cnt, res, idx = 1, x[0][1], 1
        while idx < n:
            if x[idx][0] >= res:
                cnt += 1
                res = x[idx][1]
            idx += 1
        print(cnt)