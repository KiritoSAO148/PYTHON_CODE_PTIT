if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        s1 = input()
        s2 = input()
        cnt1 = [0] * 256
        cnt2 = [0] * 256
        for x in s1: cnt1[ord(x) - 48] += 1
        for x in s2: cnt2[ord(x) - 48] += 1
        ok = True
        for x in s2:
            if cnt1[ord(x) - 48] != cnt2[ord(x) - 48]: ok = False
            if not ok: break
        print('Test ', t + 1, ': ', sep = '', end = '')
        if ok: print('YES')
        else: print('NO')