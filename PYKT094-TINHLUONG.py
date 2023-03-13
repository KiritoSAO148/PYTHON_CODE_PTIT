from math import *
import io, os, sys, time
import array as arr

class nhan_vien:
    def __init__(self, ma, ten, luong, ngay):
        self.__ma = ma
        self.__ten = ten
        self.__luong = luong
        self.__ngay = ngay

    def get_hs(self):
        nhom, nam = self.__ma[0], int(self.__ma[1:3])
        if nhom == 'A':
            if nam < 4: return 10
            if nam < 9: return 12
            if nam < 16: return 14
            return 20
        if nhom == 'B':
            if nam < 4: return 10
            if nam < 9: return 11
            if nam < 16: return 13
            return 16
        if nhom == 'C':
            if nam < 4: return 9
            if nam < 9: return 10
            if nam < 16: return 12
            return 14
        if nhom == 'D':
            if nam < 4: return 8
            if nam < 9: return 9
            if nam < 16: return 11
            return 13

    def get_ma(self):
        return self.__ma[3:]

    def get(self):
        return self.__luong * self.__ngay * self.get_hs() * 10 ** 3

    def __str__(self):
        return f'{self.__ma} {self.__ten}'

if __name__ == '__main__':
    d, a = {}, []
    for _ in range(int(input())):
        res = input().split()
        d[res[0]] = ' '.join(res[1:])
    n = int(input())
    for i in range(n): a.append(nhan_vien(input(), input(), int(input()), int(input())))
    for x in a:
        res = x.get_ma()
        print(x, end = ' ')
        print(d[res], x.get())