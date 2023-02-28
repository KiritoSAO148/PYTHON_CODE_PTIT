if __name__ == '__main__':
    TC = 10
    a = []
    while TC != 0:
        n = input().split()
        for x in n: a.append(int(x))
        TC -= len(n)
    print(len(set(list(map(lambda x : x % 42, a)))))