from math import *
import io, os, sys, time
import array as arr

class giao_vien:
    def __init__(self, id, ten, ma, d1, d2):
        self.__id = id
        self.__ten = ten
        self.__ma = ma
        self.__d1 = d1
        self.__d2 = d2

    def get_mon(self):
        c = self.__ma[0]
        if c == 'A': return 'TOAN'
        if c == 'B': return 'LY'
        return 'HOA'

    def uutien(self):
        c = self.__ma[1]
        if c == '1': return 2.0
        if c == '2': return 1.5
        if c == '3': return 1.0
        return 0

    def get_diem(self):
        return self.__d1 * 2 + self.__d2 + self.uutien()

    def get_status(self):
        return 'TRUNG TUYEN' if self.get_diem() >= 18.0 else 'LOAI'

    def __str__(self):
        return f'{self.__id} {self.__ten} {self.get_mon()} {self.get_diem()} {self.get_status()}'

if __name__ == '__main__':
    a = []
    n = int(input())
    for i in range(n):
        id = 'GV{:02d}'.format(i + 1)
        ten = input()
        ma = input()
        d1 = float(input())
        d2 = float(input())
        a.append(giao_vien(id, ten, ma, d1, d2))
    a.sort(key = lambda x : (-x.get_diem()))
    for x in a: print(x)