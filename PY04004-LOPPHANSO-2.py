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

    def __add__(self, other):
        l = self.__mau // gcd(self.__mau, other.__mau) * other.__mau
        self.__tu = l // self.__mau * self.__tu
        other.__tu = l // other.__mau * other.__tu
        self.__tu += other.__tu
        self.__mau = l
        self.rut_gon()
        return self

    def __str__(self):
        return f'{int(self.__tu)}/{int(self.__mau)}'

if __name__ == '__main__':
    a, b, c, d = list(map(int, input().split()))
    p, q = PhanSo(a, b), PhanSo(c, d)
    print(p + q)