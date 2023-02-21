from math import gcd

if __name__ == '__main__':
    TC = int(input())
    for i in range(TC):
        n = input()
        m = n[::-1]
        if gcd(int(n), int(m)) == 1: print('YES')
        else: print('NO')