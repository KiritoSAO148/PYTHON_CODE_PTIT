if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        x, y = n[:2], n[-2:]
        if x == y: print('YES')
        else: print('NO')