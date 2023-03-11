from math import *
import io, os, sys, time
import array as arr

class Tram:
    def __init__(self, name, start, end, sl):
        self.__name = name
        self.__start = start
        self.__end = end
        self.__sl = sl

    def get_name(self):
        return self.__name

    def get_start(self):
        return self.__start

    def get_end(self):
        return self.__end

    def get_sl(self):
        return self.__sl

    def time(self):
        bd, kt = int(self.__start[:2]) * 60 + int(self.__start[3:]), int(self.__end[:2]) * 60 + int(self.__end[3:])
        return kt - bd

if __name__ == '__main__':
    d1, d2 = {}, {}
    for t in range(int(sys.stdin.readline())):
        tmp = sys.stdin.readline()
        start = sys.stdin.readline()
        end = sys.stdin.readline()
        sl = int(sys.stdin.readline())
        name = ''
        for x in tmp:
            if x != '\n': name += x
        tram = Tram(name, start, end, sl)
        if tram.get_name() not in d1: d1[tram.get_name()] = tram.time()
        else: d1[tram.get_name()] += tram.time()
        if tram.get_name() not in d2: d2[tram.get_name()] = tram.get_sl()
        else: d2[tram.get_name()] += tram.get_sl()
    idx = 1
    for key in d1.keys():
        res = d2[key] / d1[key] * 60
        print('T{:02d}'.format(idx), sep = '', end = ' ')
        print(key, end = ' ')
        print('{:.2f}'.format(res))
        idx += 1