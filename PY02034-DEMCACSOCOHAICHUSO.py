if __name__ == '__main__':
    a = list(input())
    d = {}
    for i in range(int(len(a) / 2)):
        x = (ord(a[0]) - 48) * 10 + (ord(a[1]) - 48)
        if x not in d: d[x] = 1
        else: d[x] += 1
        a = a[2::]
    for x in d: print(x, d[x])