from math import *
import io, os, sys, time
from itertools import permutations

if __name__ == '__main__':
    for t in range(int(input())):
        n = int(input())
        s = ''
        for i in range(n): s += str((i + 1))
        a = list(permutations(s))
        print(len(a))
        for i in range(len(a) - 1, -1, -1):
            for x in a[i]: print(x, end = '')
            print(' ', end = '')
        print()