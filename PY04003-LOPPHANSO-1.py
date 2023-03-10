import fractions
from math import *
import io, os, sys, time
import array as arr

class PhanSo:
    def __init__(self, tu, mau):
        self.__tu = tu
        self.__mau = mau

    def rut_gon(self):
        g = gcd(self.__tu, self.__mau)
        self.__tu /= g
        self.__mau /= g
        return self

    def __str__(self):
        return f'{int(self.__tu)}/{int(self.__mau)}'

if __name__ == '__main__':
    tu, mau = map(int, input().split())
    p = PhanSo(tu, mau)
    p.rut_gon()
    print(p)