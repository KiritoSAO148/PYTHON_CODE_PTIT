if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        s = input()
        cnt = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]: cnt += 1
            else:
                print(cnt, end = '')
                print(s[i - 1], end = '')
                cnt = 1
        print(cnt, end = '')
        print(s[len(s) - 1])