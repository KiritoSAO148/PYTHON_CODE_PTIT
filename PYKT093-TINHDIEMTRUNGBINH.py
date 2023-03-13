from math import *
import io, os, sys, time
import array as arr

class sinh_vien:
    def __init__(self, ma, ten, d1, d2, d3):
        self.__ma = ma
        self.__ten = ten
        self.__d1 = d1
        self.__d2 = d2
        self.__d3 = d3

    def avg(self):
        return ceil((self.__d1 * 3 + self.__d2 * 3 + self.__d3 * 2) / 8 * 100) / 100

    def get_ma(self):
        return self.__ma

    def __str__(self):
        return self.__ma + ' ' + self.__ten + ' ' + '{:.2f}'.format(self.avg())

if __name__ == '__main__':
    a = []
    n = int(input())
    d = {}
    for i in range(n):
        ma = 'SV{:02d}'.format(i + 1)
        name = ' '.join(input().title().split())
        d1 = int(input())
        d2 = int(input())
        d3 = int(input())
        a.append(sinh_vien(ma, name, d1, d2, d3))
    for x in a:
        res = '{:.2f}'.format(x.avg())
        if res not in d: d[res] = 1
        else: d[res] += 1
    a.sort(key = lambda x : (-x.avg(), x.get_ma()))
    cnt = 1
    b = []
    for key, val in d.items(): b.append([key, val])
    b.sort(reverse = True)
    i = 0
    for x in a:
        res = '{:.2f}'.format(x.avg())
        print(x, end = ' ')
        if d[res] == 1:
            print(cnt)
            cnt += b[i][1]
            i += 1
        elif d[res] > 1:
            print(cnt)
            d[res] -= 1