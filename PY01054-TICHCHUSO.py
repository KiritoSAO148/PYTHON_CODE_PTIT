if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        s = 1
        for x in n:
            if x != '0': s *= ord(x) - 48
        print(s)