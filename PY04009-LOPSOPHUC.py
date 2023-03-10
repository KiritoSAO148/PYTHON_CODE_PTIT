from math import *
import io, os, sys, time
import array as arr

class so_phuc:
    def __init__(self, thuc, ao):
        self.__thuc = thuc
        self.__ao = ao

    def __add__(self, other):
        res = so_phuc(0, 0)
        res.__thuc = self.__thuc + other.__thuc
        res.__ao = self.__ao + other.__ao
        return res

    def __mul__(self, other):
        res = so_phuc(0, 0)
        res.__thuc = (self.__thuc * other.__thuc) - (self.__ao * other.__ao)
        res.__ao = (self.__thuc * other.__ao) + (self.__ao * other.__thuc)
        return res

    def __str__(self):
        res = ''
        res += str(self.__thuc)
        if self.__ao > 0: res += ' + '
        else: res += ' - '
        res += str(abs(self.__ao)) + 'i'
        return res

if __name__ == '__main__':
    for _ in range(int(sys.stdin.readline())):
        a, b, c, d = map(int, sys.stdin.readline().split())
        c1 = so_phuc(a, b)
        c2 = so_phuc(c, d)
        res1, res2 = (c1 + c2) * c1, (c1 + c2) * (c1 + c2)
        print(res1, ', ', res2, sep = '')