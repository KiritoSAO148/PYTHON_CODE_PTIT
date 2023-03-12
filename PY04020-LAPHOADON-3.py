from math import *
import io, os, sys, time
import array as arr

class mat_hang:
    def __init__(self, ma, ten, sl, dg, ck):
        self.__ma = ma
        self.__ten = ten
        self.__sl = sl
        self.__dg = dg
        self.__ck = ck

    def tt(self):
        return self.__sl * self.__dg - self.__ck

    def __str__(self):
        return f'{self.__ma} {self.__ten} {self.__sl} {self.__dg} {self.__ck} {self.tt()}'

if __name__ == '__main__':
    a = []
    for _ in range(int(input())): a.append(mat_hang(input(), input(), int(input()), int(input()), int(input())))
    a.sort(key = lambda x : (-x.tt()))
    for x in a: print(x)