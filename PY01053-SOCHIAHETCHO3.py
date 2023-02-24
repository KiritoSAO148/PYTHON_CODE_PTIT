if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        s = 0
        for x in n: s += ord(x) - 48
        if s % 3 == 0: print('YES')
        else: print('NO')