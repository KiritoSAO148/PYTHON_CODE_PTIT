from functools import cmp_to_key

d = {}

def cmp(a, b):
    if d[a] != d[b]: return d[b] - d[a]
    return a - b

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        d.clear()
        n = int(input())
        for _ in range(n):
            x = int(input())
            if x not in d: d[x] = 1
            else: d[x] += 1
        a = list(d)
        a.sort(key = cmp_to_key(cmp))
        print(a[0])