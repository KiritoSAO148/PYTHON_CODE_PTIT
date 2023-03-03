if __name__ == '__main__':
    a = list(input())
    s = set()
    while len(a) - 2 > 1:
        x = (ord(a[0]) - 48) * 10 + (ord(a[1]) - 48)
        s.add(x)
        a = a[2::]
    for x in sorted(list(s)): print(x, end = ' ')