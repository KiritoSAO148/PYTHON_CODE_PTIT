from math import *
import io, os, sys, time
import array as arr

class sinh_vien:
    def __init__(self, ma, ten, lop):
        self.__ma = ma
        self.__ten = ten
        self.__lop = lop

    def get_ma(self):
        return self.__ma

    def __str__(self):
        return '{} {} {}'.format(self.__ma, self.__ten, self.__lop)

def convert(s):
    res = ''
    for x in s:
        if x == 'x': res += '0'
        elif x == 'm': res += '1'
        else: res += '2'
    return res

if __name__ == '__main__':
    n = int(input())
    a, d = [], {}
    for i in range(n):
        ma = input()
        ten = input()
        lop = input()
        a.append(sinh_vien(ma, ten, lop))
    for i in range(n):
        s = input().split()
        d[s[0]] = convert(s[1])
    for i in range(n):
        print(a[i], end = ' ')
        p = 10
        for x in d[a[i].get_ma()]: p -= int(x)
        if p <= 0: p = 0
        if p == 0: print('0', 'KDDK')
        else: print(p)