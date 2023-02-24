def check(n):
    if len(n) % 2 == 0: return False
    for i in range (2, len(n), 2):
        if n[i] != n[0]: return False
    return n[0] != n[1]

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        if check(n): print('YES')
        else: print('NO')