f = [0] * 95

def fibo():
    f[1] = 1
    f[2] = 1
    for i in range(3, 95): f[i] = f[i - 1] + f[i - 2]

if __name__ == '__main__':
    fibo()
    TC = int(input())
    for t in range(TC):
        a, b = map(int, input().split())
        for i in range(a, b + 1): print(f[i], end = ' ')
        print()