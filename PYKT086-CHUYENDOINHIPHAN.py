from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    f = open('DATA.in', 'r')
    a = []
    for x in f.read().split(): a.append(x)
    for t in range(int(a[0])):
        b, s = int(a[2 * t + 1]), a[2 * t + 2]
        n, res, ans = 0, 1, ""
        for i in range(len(s) - 1, -1, -1):
            n += int(s[i]) * res
            res *= 2
        if b == 2: print(s, end = '')
        elif b == 4:
            while n > 0:
                ans += str(n % 4)
                n //= 4
        elif b == 8:
            while n > 0:
                ans += str(n % 8)
                n //= 8
        elif b == 16:
            while n > 0:
                r = n % 16
                if r <= 9: ans += str(r)
                elif r == 10: ans += 'A'
                elif r == 11: ans += 'B'
                elif r == 12: ans += 'C'
                elif r == 13: ans += 'D'
                elif r == 14: ans += 'E'
                elif r == 15: ans += 'F'
                n //= 16
        for i in range(len(ans) - 1, -1, -1): print(ans[i], end = '')
        print()