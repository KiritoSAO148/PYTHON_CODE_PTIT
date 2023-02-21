if __name__ == '__main__':
    TC = int(input())
    for i in range(TC):
        n = input()
        if n[len(n)-2:len(n)] == '86': print('YES')
        else: print('NO')