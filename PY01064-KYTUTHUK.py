def solve(n, k):
    x = pow(2, n - 1)
    if k == x: return chr(ord('A') + (n - 1))
    if k < x: return solve(n - 1, k)
    return solve(n - 1, k - x)

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n, k = map(int, input().split())
        print(solve(n, k))