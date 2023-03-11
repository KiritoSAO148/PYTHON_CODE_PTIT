from math import *
import io, os, sys, time
import array as arr

class khach_hang:
    def __init__(self, ma, name, old, new):
        self.__ma = ma
        self.__name = name
        self.__old = old
        self.__new = new

    def tinh(self):
        sl = self.__new - self.__old
        s = 0
        if sl <= 50:
            s = sl * 100
            s += s * 0.02
        elif sl <= 100:
            s = 50 * 100 + (sl - 50) * 150
            s += s * 0.03
        else:
            s = 50 * 100 + 50 * 150 + (sl - 100) * 200
            s += s * 0.05
        return s

    def __str__(self):
        return self.__ma + ' ' + self.__name + ' ' + str(round(self.tinh()))

if __name__ == '__main__':
    a = []
    for t in range(int(sys.stdin.readline())):
        ma = 'KH{:02d}'.format(t + 1)
        tmp = sys.stdin.readline()
        name = ''
        for x in tmp:
            if x != '\n': name += x
        old = int(sys.stdin.readline())
        new = int(sys.stdin.readline())
        kh = khach_hang(ma, name, old, new)
        a.append(kh)
    a.sort(key = lambda x : (-x.tinh()))
    for x in a: print(x)