from math import *
import io, os, sys, time
import array as arr
from datetime import date
from functools import cmp_to_key

class lich:
    def __init__(self, ma, mon, ten, ngay, gio, nhom):
        self.__ma = ma
        self.__mon = mon
        self.__ten = ten
        self.__ngay = ngay
        self.__gio = gio
        self.__nhom = nhom

    def get_ngay(self):
        return self.__ngay

    def get_gio(self):
        return self.__gio

    def get_mon(self):
        return self.__mon

    def __str__(self):
        return f'{self.__ma} {self.__mon} {self.__ten} {self.__ngay} {self.__gio} {self.__nhom}'

def cmp(a, b):
    date1, date2 = a.get_ngay().split('/'), b.get_ngay().split('/')
    if date1[-1] != date2[-1]: return int(date1[-1]) - int(date2[-1])
    if date1[-2] != date2[-2]: return int(date1[-2]) - int(date2[-2])
    if date1[-3] != date2[-3]: return int(date1[-3]) - int(date2[-3])
    hour1, hour2 = a.get_gio().split(':'), b.get_gio().split(':')
    if hour1[0] != hour2[0]: return int(hour1[0]) - int(hour2[0])
    if hour1[1] != hour2[1]: return int(hour1[1]) - int(hour2[1])
    return a.get_mon() < b.get_mon()

if __name__ == '__main__':
    n, m = map(int, input().split())
    d, a = {}, []
    for i in range(n):
        ma = input()
        mon = input()
        d[ma] = mon
    for i in range(m):
        b = input().split()
        a.append(lich('T{:03d}'.format(i + 1), b[0], d[b[0]], b[1], b[2], b[3]))
    a.sort(key = cmp_to_key(cmp))
    for x in a: print(x)