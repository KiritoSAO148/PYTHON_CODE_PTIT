def palind(n):
    m, tmp = n, 0
    while m != 0:
        tmp = tmp * 10 + m % 10
        m //= 10
    return tmp == n

def check(n):
    cnt = 0
    while n != 0:
        r = n % 10
        if r % 2 != 0: return False
        cnt += 1
        n //= 10
    return cnt % 2 == 0

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = int(input())
        for x in range(22, n):
            if check(x) and palind(x): print(x, end = ' ')
        print()
