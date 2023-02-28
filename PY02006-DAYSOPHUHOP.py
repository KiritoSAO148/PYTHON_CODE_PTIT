if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = int(input())
        a = list(map(int, input().split()))
        b = list(map(int, input().split()))
        ok = True
        a.sort()
        b.sort()
        for i in range(n):
            if a[i] > b[i]:
                ok = False
                break
        if ok: print('YES')
        else: print('NO')