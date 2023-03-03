if __name__ == '__main__':
    n = int(input())
    a = list(map(float, input().split()))
    s, cnt, mi, ma = 0, 0, min(a), max(a)
    for x in a:
        if x != mi and x != ma:
            s += x
            cnt += 1
    print('%.2f' % (s / cnt))