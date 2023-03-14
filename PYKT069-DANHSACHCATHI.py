from math import *
import io, os, sys, time
import array as arr
from functools import cmp_to_key

class ca_thi:
    def __init__(self, ma, ngay, gio, phong):
        self.__ma = ma
        self.__ngay = ngay
        self.__gio = gio
        self.__phong = phong

    def get_gio(self):
        return self.__gio

    def get_ngay(self):
        return self.__ngay

    def get_ma(self):
        return self.__ma

    def __str__(self):
        return f'{self.__ma} {self.__ngay} {self.__gio} {self.__phong}'

def cmp(a, b):
    date1, date2 = a.get_ngay().split('/'), b.get_ngay().split('/')
    if date1[-1] != date2[-1]: return int(date1[-1]) - int(date2[-1])
    if date1[-2] != date2[-2]: return int(date1[-2]) - int(date2[-2])
    if date1[-3] != date2[-3]: return int(date1[-3]) - int(date2[-3])
    hour1, hour2 = a.get_gio().split(':'), b.get_gio().split(':')
    if hour1[0] != hour2[0]: return int(hour1[0]) - int(hour2[0])
    if hour1[1] != hour2[1]: return int(hour1[1]) - int(hour2[1])
    return a.get_ma() < b.get_ma()

if __name__ == '__main__':
    f = open('CATHI.in', 'r')
    inp = f.read().split()
    n = int(inp[0])
    a, idx = [], 1
    for i in range(n):
        a.append(ca_thi('C{:03d}'.format(i + 1), inp[idx], inp[idx + 1], inp[idx + 2]))
        idx += 3
    for x in sorted(a, key = cmp_to_key(cmp)): print(x)