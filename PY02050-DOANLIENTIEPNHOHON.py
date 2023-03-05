from math import *
import io, os, sys, time
import array as arr

if __name__ == '__main__':
    for t in range(int(input())):
        n = int(input())
        a = list(map(int, input().split()))
        st, idx = [], [0] * n
        for i in range(n):
            while len(st) != 0 and a[i] >= a[st[-1] - 1]: st.pop()
            if len(st) == 0: idx[i] = 0
            else: idx[i] = st[-1]
            st.append(i + 1)
        for i in range(n): print((i + 1) - idx[i], end = ' ')
        print()