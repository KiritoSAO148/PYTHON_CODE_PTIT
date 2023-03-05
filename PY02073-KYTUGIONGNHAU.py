from math import *
import io, os, sys, time

if __name__ == '__main__':
    for t in range(int(input())):
        n = int(input())
        x, y, z = map(int , input().split())
        dp = [0] * 101
        dp[1] = x
        for i in range(2, n + 1):
            if i % 2 == 0: dp[i] = min(dp[i - 1] + x, dp[i // 2] + z)
            else: dp[i] = min(dp[i - 1] + x, dp[i // 2 + 1] + y + z)
        print(dp[n])