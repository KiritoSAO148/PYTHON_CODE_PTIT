def check(n):
    if len(n) < 3: return False
    l, r = 0, len(n) - 1
    while l < r and n[l] < n[l + 1]: l += 1
    while r > l and n[r] < n[r - 1]: r -= 1
    return l == r

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        if check(n): print('YES')
        else: print('NO')