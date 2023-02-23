def check(n):
    for x in n:
        if x != '1' and x != '2' and x != '0': return False
    return True

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        n = input()
        if check(n): print('YES')
        else: print('NO')