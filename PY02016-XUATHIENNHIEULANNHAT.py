from functools import cmp_to_key

d = {}

def cmp(a, b):
    if d[a] != d[b]: return d[b] - d[a]
    return a - b

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = int(input())
        a = list(map(int, input().split()))
        for x in a:
            if x not in d: d[x] = 1
            else: d[x] += 1
        a.sort(key = cmp_to_key(cmp))
        if d[a[0]] > n // 2: print(a[0])
        else: print('NO')