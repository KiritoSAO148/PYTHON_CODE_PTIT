if __name__ == '__main__':
    for t in range(int(input())):
        n, m = map(int, input().split())
        a = list(map(int, input().split()))
        idx, res = 0, max(a)
        b, c, d = [], [], []
        for i in range(n):
            if a[i] == res:
                idx = i
                break
        for i in range(n):
            if i == idx:
                d.append(m)
            d.append(a[i])
        for x in d:
            if x < 0: b.append(x)
            else: c.append(x)
        for x in b: print(x, end = ' ')
        for x in c: print(x, end = ' ')
        print()