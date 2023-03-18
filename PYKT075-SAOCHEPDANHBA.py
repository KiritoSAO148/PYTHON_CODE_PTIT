from math import *
import io, os, sys, time
import array as arr

class dien_thoai:
    def __init__(self, name, phone, date):
        self.__date = date
        self.__name = name
        self.__phone = phone

    def get_name(self):
        return self.__name[-1]

    def get(self):
        return self.__name

    def __str__(self):
        return '{}: {} {}'.format(self.__name, self.__phone, self.__date)

if __name__ == '__main__':
    f = open('SOTAY.txt', 'r')
    a = f.read().split('\n')
    res, date, i = [], '', 0
    while i < len(a) - 1:
        if a[i][:4] == 'Ngay':
            date = a[i][5:]
            i += 1
        else:
            res.append(dien_thoai(a[i], a[i + 1], date))
            i += 2
    for x in sorted(res, key = lambda x : (x.get_name(), x.get())): print(x)