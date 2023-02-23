from math import gcd

if __name__ == '__main__':
    n, k = map(int, input().split())
    cnt = 0
    for x in range(int(pow(10, k - 1)), int(pow(10, k) - 1)):
        if cnt > 9:
            cnt = 0
            print()
        if gcd(x, n) == 1:
            cnt += 1
            print(x, end = ' ')