from math import *
import io, os, sys, time
import array as arr
from sys import stdin, stdout

if __name__ == '__main__':
    for _ in range(int(stdin.readline())):
        n = int(stdin.readline())
        a = list(map(int, stdin.readline().split()))
        cnt, i = 0, 0
        while i < n - 1:
            if (max(a[i], a[i + 1])) > 2 * min(a[i], a[i + 1]):
                cnt += 1
                x = min(a[i], a[i + 1]) * 2
                a.insert(i + 1, x)
                n += 1
            else: i += 1
        stdout.write(str(cnt) + '\n')