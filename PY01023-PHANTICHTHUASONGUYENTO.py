from math import isqrt

def pt(n):
    print('1 * ', end = '')
    for i in range(2, isqrt(n) + 1):
        if n % i == 0:
            cnt = 0
            while n % i == 0:
                cnt += 1
                n //= i
            print(i, '^', cnt, sep = '', end = '')
            if n != 1: print(' * ', end = '')
    if n != 1: print(n, '^', 1, sep = '', end = '')
    print()

if __name__ == '__main__':
    TC = int(input())
    for i in range(TC):
        n = int(input())
        pt(n)