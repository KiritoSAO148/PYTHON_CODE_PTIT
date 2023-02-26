if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        s = input()
        ans = 0
        for i in range(len(s)):
            if s[i].isdigit(): ans += ord(s[i]) - 48
        s = sorted(s)
        for i in range(len(s)):
            if not s[i].isdigit(): print(s[i], end = '')
        print(ans)