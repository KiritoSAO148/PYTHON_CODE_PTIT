if __name__ == '__main__':
    n = int(input())
    a = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n): a[i] = list(map(int, input().split()))
    k = int(input())
    s1, s2 = 0, 0
    for i in range(n):
        for j in range(n):
            if j < n - i - 1: s1 += a[i][j]
            elif j > n - i - 1: s2 += a[i][j]
    if abs(s1 - s2) <= k: print('YES')
    else: print('NO')
    print(abs(s1 - s2))