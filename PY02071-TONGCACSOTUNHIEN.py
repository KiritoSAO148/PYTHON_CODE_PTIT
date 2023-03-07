from math import *
import io, os, sys, time
import array as arr

res = [] * 101
a = [] * 101
n = 0

def Try(i, sum, pos):
    for j in range(pos, 0, -1):
        a.append(j)
        if j == sum:
            tmp = a[::]
            res.append(tmp)
        elif j < sum: Try(i + 1, sum - j, j)
        a.pop()

if __name__ == '__main__':
    for t in range(int(input())):
        res.clear()
        n = int(input())
        Try(1, n, n)
        print(len(res))
        for i in range(len(res)):
            print('(', end = '')
            for j in range(len(res[i])):
                print(res[i][j], end = '')
                if j != len(res[i]) - 1: print(' ', end = '')
                else: print(') ', end = '')
        print()