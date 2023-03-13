from math import *
import io, os, sys, time
import array as arr

class thi_sinh:
    def __init__(self, id, name, p, dt, kv):
        self.__id = id
        self.__name = name
        self.__p = p
        self.__dt = dt
        self.__kv = kv

    def uutien(self):
        t = self.__kv
        if t == 1: return 1.5
        if t == 2: return 1
        return 0

    def dtoc(self):
        if self.__dt == 'Kinh': return 0
        return 1.5

    def get(self):
        return self.__p + self.uutien() + self.dtoc()

    def __str__(self):
        res = self.get()
        if res >= 20.5: s = 'Do'
        else: s = 'Truot'
        return f'{self.__id} {self.__name} {res} {s}'

if __name__ == '__main__':
    a = []
    for i in range(int(input())):
        name = ' '.join(input().title().split())
        p = float(input())
        dt = input()
        kv = int(input())
        a.append(thi_sinh('TS{:02d}'.format(i + 1), name, p, dt, kv))
    a.sort(key = lambda x : -x.get())
    for x in a: print(x)