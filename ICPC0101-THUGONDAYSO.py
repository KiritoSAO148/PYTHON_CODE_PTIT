if __name__ == '__main__':
    n = int(input())
    a = list(map(int, input().split()))
    res = []
    for x in a:
        if len(res) == 0: res.append(x)
        else:
            if (res[-1] + x) % 2 == 0: res.pop()
            else: res.append(x)
    print(len(res))