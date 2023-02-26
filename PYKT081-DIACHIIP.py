def check(s):
    for i in range(len(s)):
        if s[i].isalpha(): return False
    return True

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        a = input().split('.')
        b = []
        for x in a:
            if check(x): b.append(x)
        #print(b)
        ok = True
        if len(b) < 4: ok = False
        else:
            for x in b:
                if int(x) < 0 or int(x) > 255: ok = False
                if not ok: break
        if ok: print('YES')
        else: print('NO')