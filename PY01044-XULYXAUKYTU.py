from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    s1 = set(input().lower().split())
    s2 = set(input().lower().split())
    for x in sorted(list(s1.union(s2))): print(x, end = ' ')
    print()
    for x in sorted(list(s1.intersection(s2))): print(x, end = ' ')