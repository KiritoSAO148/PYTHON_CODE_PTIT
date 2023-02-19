from math import *

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n, x, m = map(float, input().split())
        ans = 0
        while n < m:
            n += n * x / 100
            ans += 1
        print(ans)