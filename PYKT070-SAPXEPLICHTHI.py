from math import *
import io, os, sys, time
import array as arr
from functools import cmp_to_key
from datetime import date

class lich_thi:
    def __init__(self, ma, ngay, gio, id, ten, nhom, sl):
        self.__ma = ma
        self.__ngay = ngay
        self.__gio = gio
        self.__id = id
        self.__ten = ten
        self.__nhom = nhom
        self.__sl = sl

    def get_ngay(self):
        return self.__ngay

    def get_gio(self):
        return self.__gio

    def get_ma(self):
        return self.__ma

    def __str__(self):
        return f'{self.__ngay} {self.__gio} {self.__id} {self.__ten} {self.__nhom} {self.__sl}'

if __name__ == '__main__':
    f1, f2, f3 = open('MONTHI.in', 'r'), open('CATHI.in', 'r'), open('LICHTHI.in', 'r')
    a, b, c = f1.read().split('\n'), f2.read().split('\n'), f3.read().split('\n')
    d1, d2 = {}, {}
    idx = 1
    n = int(a[0])
    for i in range(n):
        d1[a[idx]] = a[idx + 1]
        idx += 3
    m, k, idx = int(b[0]), int(c[0]), 1
    for i in range(m):
        d2['C{:03d}'.format(i + 1)] = [b[idx], b[idx + 1], b[idx + 2]]
        idx += 3
    idx = 1
    res = []
    for i in range(k):
        tmp = c[idx].split()
        ngay, gio, id = d2[tmp[0]][0], d2[tmp[0]][1], d2[tmp[0]][2]
        ten = d1[tmp[1]]
        res.append(lich_thi(tmp[0], ngay, gio, id, ten, tmp[2], tmp[3]))
        idx += 1
    for x in sorted(res, key = lambda x : (x.get_ngay(), x.get_gio())): print(x)