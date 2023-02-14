a, K, n = map(int, input().split())
if n <= a: print(-1)
else:
    res = K
    res -= a % K
    if res > n - a: print(-1)
    else:
        for x in range (res, n - a + 1, K): print(x, end = ' ')