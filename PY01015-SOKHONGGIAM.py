def check(n):
    for i in range(len(n) - 1):
        if ord(n[i]) > ord(n[i + 1]): return False
    return True

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        if check(n): print('YES')
        else: print('NO')