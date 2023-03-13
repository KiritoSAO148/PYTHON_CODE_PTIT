from datetime import datetime, date
from math import *
import io, os, sys, time
import array as arr

class film:
    def __init__(self, id, ma, ngay, ten, sl, tl):
        self.__id = id
        self.__ma = ma
        self.__ten = ten
        self.__ngay = ngay
        self.__sl = sl
        self.__tl = tl

    def get_sl(self):
        return self.__sl

    def get_ten(self):
        return self.__ten

    def get_ngay(self):
        a = list(map(int, self.__ngay.split('/')))
        return date(a[-1], a[-2], a[-3]).day

    def __str__(self):
        return f'{self.__id} {self.__tl} {self.__ngay} {self.__ten} {self.__sl}'

if __name__ == '__main__':
    n, m = map(int, input().split())
    d, a = {}, []
    for i in range(n):
        s = input()
        d['TL{:03d}'.format(i + 1)] = s
    for i in range(m):
        id = 'P{:03d}'.format(i + 1)
        ma = input()
        a.append(film(id, ma, input(), input(), int(input()), d[ma]))
    a.sort(key = lambda x : (-x.get_ngay(), x.get_ten(), -x.get_sl()))
    for x in a: print(x)