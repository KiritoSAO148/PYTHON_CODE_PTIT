if __name__ == '__main__':
    n, m = map(int, input().split())
    a = set(list(map(int, input().split())))
    b = set(list(map(int, input().split())))
    if a == b: print('YES')
    else: print('NO')