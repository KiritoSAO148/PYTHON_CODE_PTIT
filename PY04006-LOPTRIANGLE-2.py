from math import *
import io, os, sys, time
import array as arr

class Point:
    def __init__(self, x, y):
        self.__x = x
        self.__y = y

    def distance(self, another):
        return sqrt((self.__x - another.__x) ** 2 + (self.__y - another.__y) ** 2)

class Triangle:
    def __init__(self, a, b, c):
        self.__a = a
        self.__b = b
        self.__c = c

    def area(self):
        x, y, z = self.__a, self.__b, self.__c
        return sqrt((x + y + z) * (x + y - z) * (x - y + z) * (y + z - x)) / 4

if __name__ == '__main__':
    res = []
    TC = int(sys.stdin.readline())
    for t in range(TC):
        res += [float(x) for x in sys.stdin.readline().split()]
    i = 0
    for _ in range(TC):
        a, b, c, d, e, f = res[i], res[i + 1], res[i + 2], res[i + 3], res[i + 4], res[i + 5]
        x, y, z = Point(a, b), Point(c, d), Point(e, f)
        n, m, k = x.distance(y), x.distance(z), y.distance(z)
        if n > 0 and m > 0 and k > 0 and n + m > k and n + k > m and m + k > n:
            triangle = Triangle(n, m, k)
            print('{:.2f}'.format(triangle.area()))
        else:
            print('INVALID')
        i += 6