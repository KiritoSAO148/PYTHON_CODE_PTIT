from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    for t in range(int(sys.stdin.readline())):
        n = int(sys.stdin.readline())
        s = ''
        while n >= 10:
            r = n % 10
            n //= 10
            if r < 5: s += '0'
            else:
                s += '0'
                n += 1
        reversed(s)
        print(str(n) + s)