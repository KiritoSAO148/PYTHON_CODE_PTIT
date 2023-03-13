from math import *
import io, os, sys, time
import array as arr

class mon:
    def __init__(self, ma, ten, ht):
        self.__ma = ma
        self.__ten = ten
        self.__ht = ht

    def get_ma(self):
        return self.__ma

    def __str__(self):
        return f'{self.__ma} {self.__ten} {self.__ht}'

if __name__ == '__main__':
    a = []
    for _ in range(int(input())): a.append(mon(input(), input(), input()))
    for x in sorted(a, key = lambda x : x.get_ma()): print(x)