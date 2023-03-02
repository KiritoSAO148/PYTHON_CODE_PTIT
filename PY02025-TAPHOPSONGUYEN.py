if __name__ == '__main__':
    n, m = map(int, input().split())
    a = set(list(map(int, input().split())))
    b = set(list(map(int, input().split())))
    inter = a.intersection(b)
    res1 = a - b
    res2 = b - a
    for x in sorted(inter): print(x, end = ' ')
    print()
    for x in sorted(res1): print(x, end = ' ')
    print()
    for x in sorted(list(res2)): print(x, end = ' ')
    print()
