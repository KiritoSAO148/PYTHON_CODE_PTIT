from math import *

def prime(n):
    if n < 2: return False
    for i in range(2, isqrt(n) + 1):
        if n % i == 0: return False
    return True

def check(n):
    for i in range(len(n)):
        if prime(i) and (n[i] != '2' and n[i] != '3' and n[i] != '5' and n[i] != '7'): return False
        if not prime(i) and (n[i] == '2' or n[i] == '3' or n[i] == '5' or n[i] == '7'): return False
    return True

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        if check(n): print('YES')
        else: print('NO')