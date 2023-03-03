if __name__ == '__main__':
    n, m = map(int, input().split())
    a = [[0 for _ in range(m)] for _ in range(n)]
    mi, ma = 10 ** 18, -10 ** 18
    for i in range(n): a[i] = list(map(int, input().split()))
    for i in range(n):
        for j in range(m):
            mi = min(mi, a[i][j])
            ma = max(ma, a[i][j])
    ok = 0
    res = abs(ma - mi)
    for i in range(n):
        for j in range(m):
            if a[i][j] == res:
                ok = 1
                break
    if not ok: print('NOT FOUND')
    else:
        print(res)
        for i in range(n):
            for j in range(m):
                if a[i][j] == res:
                    print('Vi tri ', '[', i, ']', '[', j, ']', sep='')
                    ok = 1