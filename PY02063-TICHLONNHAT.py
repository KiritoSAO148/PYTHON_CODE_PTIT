if __name__ == '__main__':
    n = int(input())
    a = sorted((list(map(int, input().split()))))
    res1, res2, res3, res4, res5, res6 = -10 ** 18, -10 ** 18,-10 ** 18, -10 ** 18, -10 ** 18, -10 ** 18
    res1 = max(res1, a[0] * a[1])
    res2 = max(res2, a[n - 2] * a[n - 1])
    res3 = max(res3, a[0] * a[1] * a[n - 1])
    res4 = max(res4, a[0] * a[n - 1] * a[n - 2])
    res5 = max(res5, a[0] * a[1] * a[2])
    res6 = max(res6, a[n - 3] * a[n - 2] * a[n - 1])
    print(max(res1, res2, res3, res4, res5, res6))