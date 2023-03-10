from math import *
import io, os, sys, time
import array as arr

class thisinh:
    def __init__(self, name, date, d1, d2, d3):
        self.__name = name
        self.__date = date
        self.__d1 = d1
        self.__d2 = d2
        self.__d3 = d3

    def __str__(self):
        return self.__name + ' ' + self.__date + ' ' + ('%.1f' % (self.__d1 + self.__d2 + self.__d3))

if __name__ == '__main__':
    a = []
    for line in sys.stdin: a += line.split('\n')
    for x in a:
        if x == '': a.remove(x)
    ts = thisinh(a[0], a[1], float(a[2]), float(a[3]), float(a[4]))
    print(ts)