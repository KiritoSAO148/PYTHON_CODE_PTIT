def Try(n, a, b, c):
    if n == 1: print(a, ' -> ', c, sep = '')
    else:
        Try(n - 1, a, c, b)
        print(a, ' -> ', c, sep = '')
        Try(n - 1, b, a, c)

if __name__ == '__main__':
    Try(int(input()), 'A', 'B', 'C')