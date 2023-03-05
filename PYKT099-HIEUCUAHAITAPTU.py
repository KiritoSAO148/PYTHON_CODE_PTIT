from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    f1 = open('DATA1.in', 'r')
    f2 = open('DATA2.in', 'r')
    a, b = f1.read().split(), f2.read().split()
    for i in range(len(a)): a[i] = a[i].lower()
    for i in range(len(b)): b[i] = b[i].lower()
    a = set(a)
    b = set(b)
    for x in sorted(list(a - b)): print(x, end = ' ')
    print()
    for x in sorted(list(b - a)): print(x, end = ' ')
    f1.close()
    f2.close()