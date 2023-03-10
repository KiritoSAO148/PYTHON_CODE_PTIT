from math import *
import io, os, sys, time
import array as arr

class Rectangle:
    def __init__(self, x, y, c):
        self.__x = x
        self.__y = y
        self.__c = c

    def perimeter(self):
        return (self.__x + self.__y) * 2

    def area(self):
        return self.__x * self.__y

    def color(self):
        return self.__c.title()

arr = input().split()
if int(arr[0]) > 0 and int(arr[1]) > 0:
    r = Rectangle(int(arr[0]), int(arr[1]), arr[2])
    print('{} {} {}'.format(r.perimeter(), r.area(), r.color()))
else: print('INVALID')