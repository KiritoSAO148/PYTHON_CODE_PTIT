if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        s = input()
        a = []
        for x in s:
            if x.isalpha(): a.append(x)
            elif x.isdigit():
                c = a.pop(0)
                for i in range(int(x)): print(c, sep = '', end = '')
        print()