if __name__ == '__main__':
    s = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    TC = int(input())
    for t in range(TC):
        n, b = map(int, input().split())
        ans = ''
        while n != 0:
            ans = s[n % b] + ans
            n //= b
        print(ans)