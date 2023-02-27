if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        s = input()
        ans, res = 0, 10 ** 500
        for i in range(len(s)):
            if s[i].isdigit(): ans = ans * 10 + (ord(s[i]) - 48)
            else:
                if ans < res and ans != 0: res = ans
                ans = 0
        if ans < res and ans != 0: res = ans
        print(res)