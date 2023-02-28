from math import isqrt

p = [True] * 1000001

def sieve():
    p[0] = p[1] = False
    for i in range(2, isqrt(1000000) + 1):
        if p[i]:
            for j in range(i * i, 1000001, i): p[j] = False

if __name__ == '__main__':
    sieve()
    n = int(input())
    a = list(map(int, input().split()))
    cnt = [0] * 1000001
    for x in a:
        if p[x]: cnt[x] += 1
    for x in a:
        if cnt[x] > 0:
            print(x, cnt[x])
            cnt[x] = 0