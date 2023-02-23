if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        if n[:2] == n[-2:]: print('YES')
        else: print('NO')