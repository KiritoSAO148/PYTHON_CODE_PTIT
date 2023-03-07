from math import *
import io, os, time
import array as arr
from sys import stdin
import re

if __name__ == '__main__':
    s = ""
    regex = "[\w\s,:]+"
    while True:
        try: s += input()
        except EOFError: break
    s = re.findall(regex, s)
    for x in s:
        res = x.lower().split()
        res[0] = res[0].title()
        print(' '.join(res))