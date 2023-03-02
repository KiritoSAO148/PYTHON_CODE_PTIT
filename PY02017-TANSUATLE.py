if __name__ == '__main__':
    for t in range(int(input())):
        n = int(input())
        a = list(map(int, input().split()))
        cnt = [0] * 1000001
        for x in a: cnt[x] += 1
        for i in range(min(a), max(a) + 1):
            if cnt[i] %2 == 1:
                print(i)
                break