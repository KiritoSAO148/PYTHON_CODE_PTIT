def check(n):
    while n != 0:
        r = n % 10;
        if r != 4 and r != 7: return False
        n //= 10
    return True

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = int(input())
        if check(n): print('YES')
        else: print('NO')