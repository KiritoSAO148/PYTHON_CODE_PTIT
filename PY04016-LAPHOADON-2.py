from math import *
import io, os, sys, time
import array as arr
from datetime import date

class khach_hang:
    def __init__(self, ma, name, id, vao, ra, service):
        self.__ma = ma
        self.__name = name
        self.__id = id
        self.__vao = vao
        self.__ra = ra
        self.__service = service

    def get_value(self):
        c = self.__id[0]
        if c == '1': return 25
        if c == '2': return 34
        if c == '3': return 50
        return 80

    def get_day(self):
        a, b = list(map(int, self.__vao.split('/'))), list(map(int, self.__ra.split('/')))
        return abs(date(b[-1], b[-2], b[-3]) - date(a[-1], a[-2], a[-3])).days + 1

    def get(self):
        return (self.get_day() * self.get_value()) + self.__service

    def __str__(self):
        return '{} {} {} {} {}'.format(self.__ma, self.__name, self.__id, self.get_day(), self.get())

if __name__ == '__main__':
    a = []
    n = int(input())
    for i in range(n):
        ma = 'KH{:02d}'.format(i + 1)
        name = input()
        id = input()
        vao = input()
        ra = input()
        service = int(input())
        a.append(khach_hang(ma, name, id, vao, ra, service))
    a.sort(key = lambda x : (-x.get()))
    for x in a: print(x)