from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    f1 = open('DATA1.in', 'r')
    f2 = open('DATA2.in', 'r')
    a, b = set(f1.read().lower().split()), set(f2.read().lower().split())
    for x in sorted(list(a - b)): print(x, end = ' ')
    print()
    for x in sorted(list(b - a)): print(x, end = ' ')
    f1.close()
    f2.close()