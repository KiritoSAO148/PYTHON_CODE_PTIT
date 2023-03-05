from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    lst = []
    for _ in range(int(input())):
        s = input() + '@'
        res = ''
        for i in range(len(s)):
            if s[i].isdigit(): res += s[i]
            else:
                if res != '': lst.append(int(res))
                res = ''
    for x in sorted(lst): print(x)