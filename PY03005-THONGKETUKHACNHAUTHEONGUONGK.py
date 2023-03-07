from math import *
import io, os, sys, time
import array as arr
from re import findall
from collections import Counter

if __name__ == '__main__':
    d = set()
    d.add(' ')
    for i in range(48, 58): d.add(chr(i))
    for i in range(65, 91): d.add(chr(i))
    for i in range(97, 123): d.add(chr(i))
    a = []
    n, k = map(int, input().split())
    for t in range(n):
        s = input().lower()
        res = ''
        for x in s:
            if x in d: res += x
            elif x not in d: res += ' '
        for x in res.split(): a.append(x)
    counter = list(Counter(a).items())
    counter.sort(key = lambda x : (-x[1], x[0]))
    for x in counter:
        if x[1] >= k: print(x[0], x[1])