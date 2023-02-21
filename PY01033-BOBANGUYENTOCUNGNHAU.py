from math import gcd

if __name__ == '__main__':
    a, b = map(int, input().split())
    for i in range(a, b - 1):
        for j in range(i + 1, b):
            for k in range(j + 1, b + 1):
                if gcd(i, j) == 1 and gcd(i, k) == 1 and gcd(j, k) == 1: print('(', i, ', ', j, ', ', k, ')', sep = '')