n = int(input())
cnt4, cnt7 = 0, 0
while n > 0:
    r = n % 10
    if r == 4: cnt4 += 1
    elif r == 7: cnt7 += 1
    n //= 10
if cnt4 + cnt7 == 4 or cnt4 + cnt7 == 7: print('YES')
else: print('NO')