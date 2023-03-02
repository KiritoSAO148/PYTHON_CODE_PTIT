if __name__ == '__main__':
    n = int(input())
    a = []
    while True:
        tmp = list(map(int, input().split()))
        a += tmp
        if len(a) == n: break
    le = [x for x in a if x % 2 == 1]
    chan = [x for x in a if x % 2 == 0]
    le.sort()
    chan.sort(reverse = True)
    mark = [0] * n
    for i in range(n):
        if a[i] % 2 == 1: mark[i] = 1
    for i in range(n):
        if mark[i]:
            print(le[-1], end = ' ')
            le.pop()
        else:
            print(chan[-1], end = ' ')
            chan.pop()