if __name__ == '__main__':
    a = list(input())
    lst, d = [], {}
    while len(a) - 2 > 1:
        x = (ord(a[0]) - 48) * 10 + (ord(a[1]) - 48)
        if x not in d:
            lst.append(x)
            d[x] = 1
        a = a[2::]
    for x in lst: print(x, end = ' ')