from math import *
import io, os, sys, time
import array as arr

class sinh_vien:
    def __init__(self, id, name, team):
        self.__id = id
        self.__name = name
        self.__team = team

    def get_name(self):
        return self.__name

    def get_team(self):
        return self.__team

    def __str__(self):
        return f'{self.__id} {self.__name}'

if __name__ == '__main__':
    m = int(input())
    a, d = [], {}
    for i in range(m):
        ma = input()
        ten = input()
        d['Team{:02d}'.format(i + 1)] = (ma, ten)
    n = int(input())
    for i in range(n): a.append(sinh_vien('C{:03d}'.format(i + 1), input(), input()))
    a.sort(key = lambda x : x.get_name())
    for x in a:
        print(x, end = ' ')
        for y in d[x.get_team()]: print(y, end = ' ')
        print()