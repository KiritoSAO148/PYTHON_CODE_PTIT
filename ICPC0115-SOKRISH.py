import math


def krish(n):
    s, m = 0, n
    while n != 0:
        s += math.factorial(n % 10)
        n //= 10
    return s == m

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = int(input())
        if krish(n): print('Yes')
        else: print('No')