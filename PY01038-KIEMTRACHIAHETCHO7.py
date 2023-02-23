if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = int(input())
        cnt, ok = 1, 1
        while n % 7 != 0 and cnt <= 1000:
            n += int(str(n)[::-1])
            cnt += 1
            if cnt == 1001:
                ok = 0
                break
        if ok == 0: print(-1)
        else: print(n)