from math import *
import io, os, sys, time

if __name__ == '__main__':
    f = open('CONTACT.in', 'r')
    a = []
    a.append(f.read().split('\n'))
    s = set()
    for x in a[0]: s.add(x.lower())
    s = sorted(list(s))
    for x in s: print(x)