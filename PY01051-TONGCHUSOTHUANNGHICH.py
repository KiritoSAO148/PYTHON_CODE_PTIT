if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = int(input())
        s = 0
        while n > 0:
            s += n % 10
            n //= 10
        if s >= 10 and s == int(str(s)[::-1]): print('YES')
        else: print('NO')