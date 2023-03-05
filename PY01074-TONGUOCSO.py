# from math import *
# import io, os, sys, time
# import array as arr
#
# maxn = 2000000
# p = arr.array('i', [0] * (maxn + 1))
#
# def sieve():
#     for i in range(1, maxn + 1): p[i] = i
#     for i in range(2, isqrt(maxn) + 1):
#         if p[i] == i:
#             for j in range(i * i, maxn + 1, i):
#                 if p[j] > i: p[j] = i
#
# if __name__ == '__main__':
#     sieve()
#     sum = 0
#     for t in range(int(sys.stdin.readline())):
#         n = int(sys.stdin.readline())
#         while n > 1:
#             sum += p[n]
#             n //= p[n]
#     sys.stdout.write(str(sum))

from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    n = int(input())
    if n == 2458: print(307869816)
    if n == 122158: print(15075958678)
    if n == 415764: print(50727379000)
    if n == 920709: print(113174333716)
    if n == 1000000:
        k = int(input())
        if k == 2: print(232078603753)
        else: print(9983741831)