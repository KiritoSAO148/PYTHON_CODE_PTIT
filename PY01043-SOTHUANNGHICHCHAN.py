def check(n):
    if n != n[::-1]: return False
    for x in n:
        if (ord(x) - 48) % 2 == 1: return False
    if len(n) % 2 == 1: return False
    return True

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        for i in range(22, int(n)):
            if check(str(i)): print(i, end = ' ')
        print()