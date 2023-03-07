from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    for t in range(int(input())):
        s = input()
        cnt, idx = 0, 0
        for i in range(len(s)):
            if cnt == 100:
                idx = i
                break
            cnt += 1
        if len(s) <= 100: print(s)
        else:
            if s[idx] != '' or s[idx] != ' ':
                while s[idx - 1] != '' and s[idx - 1] != ' ': idx -= 1
            for i in range(idx): print(s[i], end = '')
            print()