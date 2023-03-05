import gc
from os import remove
import re

t = int(input())
while t > 0:
    t -= 1
    n = input()
    inLish = ' ' + input().replace(' ', '  ')+' '
    arr = []
    i = -18
    while i < 19 and len(arr) < 4:
        if i < 0:
            str = '-' + '\d'*abs(i) + ' '
        elif i > 0:
            str = ' ' + '\d'*abs(i) + ' '
        else:
            i += 1
            continue
        tmpArr = re.findall(str, inLish)
        arr += [int(x) for x in tmpArr]
        i += 1
    sum = min(arr)
    for k in range(2):
        arr.remove(min(arr))
        sum += min(arr)
    print(sum)