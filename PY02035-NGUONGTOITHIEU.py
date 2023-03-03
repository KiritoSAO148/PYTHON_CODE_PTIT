if __name__ == '__main__':
    a = input()
    k = int(input())
    lst, d = [], {}
    for i in range(len(a) // 2):
        x = int(a[0]) * 10 + int(a[1])
        if x not in d:
            d[x] = 1
            lst.append(x)
        else: d[x] += 1
        a = a[2:]
    ok = 0
    for x in sorted(lst):
        if d[x] >= k:
            print(x, d[x])
            ok = 1
    if not ok: print('NOT FOUND')