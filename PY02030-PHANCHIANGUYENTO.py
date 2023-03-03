from math import isqrt

def prime(n):
    if n < 2: return False
    for i in range(2, isqrt(n) + 1):
        if n % i == 0: return False
    return True

if __name__ == '__main__':
    n = int(input())
    a = list(map(int, input().split()))
    b, d = [], {}
    for x in a:
        if x not in d:
            b.append(x)
            d[x] = 1
    idx, m = -1, len(b)
    for i in range(m):
        if prime(sum(b[:i+1])) and prime(sum(b[i+1:])):
            idx = i
            break
    if idx == -1: print('NOT FOUND')
    else: print(idx)