def Try (s, a, b, c, n):
    if len(s) == n and a > 0 and b > 0 and c > 0 and a <= b and b <= c: print(s)
    if len(s) < n:
        Try(s + 'A', a + 1, b, c, n)
        Try(s + 'B', a, b + 1, c, n)
        Try(s + 'C', a, b, c + 1, n)

if __name__ == '__main__':
    n = int(input())
    for i in range(3, n + 1): Try('', 0, 0, 0, i)