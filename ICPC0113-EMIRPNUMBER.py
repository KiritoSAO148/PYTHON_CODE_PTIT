from math import isqrt
p = [1] * 1000001

def sieve():
    p[0] = p[1] = 0
    for i in range(2, isqrt(1000000) + 1):
        if p[i]:
            for j in range(i * i, 1000001, i): p[j] = 0

if __name__ == '__main__':
    TC = int(input())
    sieve()
    for t in range(TC):
        n = input()
        d = {}
        for i in range(13, int(n)):
            j = str(i)[::-1]
            if int(j) != i and p[int(i)] and p[int(j)] and int(j) < int(n) and int(i) not in d and int(j) not in d:
                print(int(i), int(j), end = ' ')
                d[int(i)] = 1
                d[int(j)] = 1
        print()