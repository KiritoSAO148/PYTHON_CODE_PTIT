from functools import cmp_to_key

def sum(n):
    s = 0
    while n != 0:
        s += n % 10
        n //= 10
    return s

def cmp(a, b):
    if sum(a) != sum(b): return sum(a) - sum(b)
    return a - b

if __name__ == '__main__':
    for t in range(int(input())):
        n = int(input())
        a = list(map(int, input().split()))
        a.sort(key = cmp_to_key(cmp))
        for x in a: print(x, end = ' ')
        print()