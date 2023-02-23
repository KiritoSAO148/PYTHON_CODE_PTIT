import math

def prime(n):
    if n < 2: return False
    for i in range(2, math.isqrt(n) + 1):
        if n % i == 0: return False
    return True

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        if prime(int(n[len(n) - 4:len(n)])): print('YES')
        else: print('NO')