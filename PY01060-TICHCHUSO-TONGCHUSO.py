if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        s, p = 0, 1
        for i in range(len(n)):
            if i % 2 == 1: s += (ord(n[i]) - 48)
            elif n[i] != '0': p *= (ord(n[i]) - 48)
        print(p, s)