from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    for _ in range(int(input())):
        s = input()
        t = ''
        for x in s:
            if x == '(' or x == ')': t += x
        s = t[::]
        st, a, cnt = [], [0] * len(s), 0
        for i in range(len(s)):
            if s[i] == '(' or len(st) == 0:
                cnt += 1
                a[i] = cnt
                st.append(i)
            elif s[i] == ')':
                a[i] = a[st[-1]]
                st.pop()
        for x in a: print(x, end = ' ')
        print()