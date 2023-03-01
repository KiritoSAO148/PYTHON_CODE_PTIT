from functools import cmp_to_key

def mul(n):
    ans = 1
    while n != 0:
        ans *= n % 10
        n //= 10
    return ans

def cmp(a, b):
    if mul(a) != mul(b): return mul(a) - mul(b)
    return a - b

if __name__ == '__main__':
    for t in range(int(input())):
        n = int(input())
        a = list(map(int, input().split()))
        a.sort(key = cmp_to_key(cmp))
        for x in a: print(x, end = ' ')
        print()