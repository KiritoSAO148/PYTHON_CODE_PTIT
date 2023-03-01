if __name__ == '__main__':
    for t in range(int(input())):
        n = int(input())
        a = sorted(set(list(map(int, input().split()))))
        print(max(a) - min(a) - len(a) + 1)