from math import *
import io, os, sys, time
import array as arr

def check(n):
    for i in range(len(n)):
        if not n[i].isalpha(): return False
    return True

if __name__ == '__main__':
    f = open('DATA.in', 'r')
    a = []
    for x in f.read().split():
        if len(x) > 9: a.append(x)
        elif check(x): a.append(x)
    for x in sorted(a): print(x, end = ' ')