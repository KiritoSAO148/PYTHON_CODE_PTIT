def check(n):
    for i in range(1,len(n)):
        if abs(ord(n[i]) - ord(n[i - 1])) != 2: return False
    sum = 0
    for i in range(len(n)):
        sum += (ord(n[i]) - 48)
    return sum % 10 == 0

if __name__ == '__main__':
    TC = int(input())
    for i in range(TC):
        n = input()
        if check(n): print('YES')
        else: print('NO')