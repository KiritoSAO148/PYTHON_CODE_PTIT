def check(n):
    if len(set(n)) != 2: return False
    for i in range(0, len(n) - 2):
        if n[i] != n[i + 2]: return False
    return True

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        if check(n): print('YES')
        else: print('NO')