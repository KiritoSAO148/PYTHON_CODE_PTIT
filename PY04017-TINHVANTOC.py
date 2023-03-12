from math import *
import io, os, sys, time
import array as arr

class person:
    def __init__(self, ma, name, lo, time):
        self.__ma = ma
        self.__name = name
        self.__lo = lo
        self.__time = time

    def get(self):
        res = list(map(int, self.__time.split(':')))
        bd, kt = 6 * 60, res[0] * 60 + res[1]
        return 120 / (kt - bd) * 60

    def __str__(self):
        return '{} {} {} {} Km/h'.format(self.__ma, self.__name, self.__lo, round(self.get()))

if __name__ == '__main__':
    a = []
    n = int(input())
    for i in range(n):
        name = input()
        lo = input()
        res = ''
        for x in lo.split(): res += x[0].upper()
        for x in name.split(): res += x[0].upper()
        time = input()
        a.append(person(res, name, lo, time))
    a.sort(key = lambda x : (-x.get()))
    for x in a: print(x)