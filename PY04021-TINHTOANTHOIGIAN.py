from math import *
import io, os, sys, time
import array as arr

class account:
    def __init__(self, id, name, vao, ra):
        self.__id = id
        self.__name = name
        self.__vao = vao
        self.__ra = ra

    def time(self):
        return (int(self.__ra[:2]) * 60 + int(self.__ra[3:])) - (int(self.__vao[:2]) * 60 + int(self.__vao[3:]))

    def __str__(self):
        return '{} {} {} gio {} phut'.format(self.__id, self.__name, self.time() // 60, self.time() % 60)

if __name__ == '__main__':
    a = []
    for t in range(int(input())):
        id = input()
        name = input()
        vao = input()
        ra = input()
        a.append(account(id, name, vao, ra))
    a.sort(key = lambda x : (-x.time()))
    for x in a: print(x)