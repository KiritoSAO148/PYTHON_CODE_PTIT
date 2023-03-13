from math import *
import io, os, sys, time
import array as arr

class gia_dinh:
    def __init__(self, id, ten, loai, dau, cuoi):
        self.__id = id
        self.__ten = ten
        self.__loai = loai
        self.__dau = dau
        self.__cuoi = cuoi

    def get_dm(self):
        if self.__loai == 'A': return 100
        if self.__loai == 'B': return 500
        return 200

    def get_in_dm(self):
        so_dien = self.__cuoi - self.__dau
        if so_dien < self.get_dm(): return so_dien * 450
        return self.get_dm() * 450

    def get_out_dm(self):
        so_dien = self.__cuoi - self.__dau
        if so_dien > self.get_dm(): return (so_dien - self.get_dm()) * 1000
        return 0

    def thue(self):
        return self.get_out_dm() // 20

    def get(self):
        return self.get_in_dm() + self.get_out_dm() + self.thue()

    def __str__(self):
        return f'{self.__id} {self.__ten} {self.get_in_dm()} {self.get_out_dm()} {self.thue()} {self.get()}'

if __name__ == '__main__':
    a = []
    for i in range(int(input())):
        id = 'KH{:02d}'.format(i + 1)
        name = ' '.join(input().title().split())
        loai, dau, cuoi = input().split()
        a.append(gia_dinh(id, name, loai, int(dau), int(cuoi)))
    a.sort(key = lambda x : -x.get())
    for x in a: print(x)