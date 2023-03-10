from math import *
import io, os, sys, time
import array as arr

class Matrix:
    def __init__(self, a, n, m):
        self.__a = a
        self.__n = n
        self.__m = m

    def __mul__(self, other):
        res = [[0 for _ in range(self.__n)] for _ in range(self.__n)]
        for i in range(self.__n):
            for j in range(self.__n):
                for k in range(self.__m): res[i][j] += self.__a[i][k] * other.__a[k][j]
        return res


if __name__ == '__main__':
    for t in range(int(sys.stdin.readline())):
        n, m = map(int, sys.stdin.readline().split())
        a = []
        for i in range(n): a.append(list(map(int, sys.stdin.readline().split())))
        matrix1 = Matrix(a, n, m)
        b = [list(x) for x in zip(*a)]
        matrix2 = Matrix(b, m, n)
        res = matrix1.__mul__(matrix2)
        for x in res:
            for y in x: print(y, end = ' ')
            print()