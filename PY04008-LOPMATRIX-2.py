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
    a = []
    for line in sys.stdin: a += map(int, line.split())
    idx = 1
    for _ in range(a[0]):
        n, m = a[idx], a[idx + 1]
        idx += 2
        b = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                b[i][j] = a[idx]
                idx += 1
        matrix1 = Matrix(b, n, m)
        c = [list(x) for x in zip(*b)]
        matrix2 = Matrix(c, m, n)
        res = matrix1.__mul__(matrix2)
        for x in res:
            for y in x: print(y, end=' ')
            print()