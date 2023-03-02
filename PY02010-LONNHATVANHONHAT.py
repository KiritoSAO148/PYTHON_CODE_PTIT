if __name__ == '__main__':
    a = []
    while True:
        n = int(input())
        if n == 0: break
        res = []
        for _ in range(n):
            x = int(input())
            res.append(x)
        a.append([min(res), max(res)])
    for x in a:
        if x[0] == x[1]: print('BANG NHAU')
        else: print(x[0], x[1])