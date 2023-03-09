from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    n = int(input())
    a = []
    while True:
        try:
            s = input().split()
            for x in s: a.append(int(x))
        except EOFError: break
    check = 0
    s, ma = set(), a[-1]
    for x in a: s.add(x)
    for x in range(1, ma + 1):
        if x not in s:
            print(x)
            check = 1
    if not check: print('Excellent!')