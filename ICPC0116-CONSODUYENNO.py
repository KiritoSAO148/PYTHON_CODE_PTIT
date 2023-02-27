if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        if n[0] == n[-1]:
            print('YES')
        else:
            print('NO')