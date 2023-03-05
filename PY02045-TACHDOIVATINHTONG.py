from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    n = input()
    l = len(n)
    while l > 1:
        n = str(int(n[:l//2]) + int(n[l//2:]))
        print(n)
        l = len(n)