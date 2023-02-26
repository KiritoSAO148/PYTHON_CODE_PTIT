def solve(n):
    n = str(n)
    while len(n) % 3 != 0:
        n = '0' + n
    list = [n[i:i + 3] for i in range(0, len(n), 3)]
    ans = ''
    for x in list:
        ans += str(int(x, 2))
    return ans

if __name__ == '__main__':
    print(solve(input()))