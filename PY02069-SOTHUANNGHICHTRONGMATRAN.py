def palind(n): return n >= 10 and int(str(n)[::-1]) == n

if __name__ == '__main__':
    n, m = map(int, input().split())
    a = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n): a[i] = list(map(int, input().split()))
    res = -10**18
    for i in range(n):
        for j in range(m):
            if palind(a[i][j]): res = max(res, a[i][j])
    if (res == -10 ** 18): print('NOT FOUND')
    else:
        print(res)
        for i in range(n):
            for j in range(m):
                if a[i][j] == res: print('Vi tri ', '[', i, ']', '[', j, ']', sep='')