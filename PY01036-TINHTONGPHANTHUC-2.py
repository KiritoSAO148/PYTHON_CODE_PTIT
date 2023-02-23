if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = int(input())
        S = 0
        if n % 2 == 1:
            for i in range(1, n + 1, 2): S += 1 / i
            print('%.6f' % S)
        else:
            for i in range(2, n + 1, 2): S += 1 / i
            print('%.6f' % S)