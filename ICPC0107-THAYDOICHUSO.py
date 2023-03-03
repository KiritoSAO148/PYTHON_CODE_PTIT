if __name__ == '__main__':
    for t in range(int(input())):
        n, k = map(int, input().split())
        a = list(input().split())
        if len(a) > 1: str1, str2 = a[0], a[1]
        else:
            str1, str2 = a[0], input()
        p, q = str(min(n, k)), str(max(n, k))
        res1 = int(str1.replace(q, p)) + int(str2.replace(q, p))
        res2 = int(str1.replace(p, q)) + int(str2.replace(p, q))
        print(res1, res2)