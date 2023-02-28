if __name__ == '__main__':
    n = int(input())
    a = list(map(int, input().split()))
    a.sort()
    ok = False
    for i in range(1, n):
        if a[i] != a[i - 1] and a[i] != a[i - 1] + 1:
            print(a[i - 1] + 1)
            ok = True
            break
    if not ok: print(max(a) + 1)