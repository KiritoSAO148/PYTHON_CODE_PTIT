def check(n):
    if n[0] != '6': return False
    for i in range(len(n)):
        if n[i] != '6' and n[i] != '8': return False
        if i >= 2 and n[i - 2:i + 1] == '888': return False
    return True

if __name__ == '__main__':
    n = input()
    if check(n): print('YES')
    else: print('NO')