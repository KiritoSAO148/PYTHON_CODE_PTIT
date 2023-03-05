t = int(input())
for j in range(t):
    k = int(input())
    n = input()+" "
    res = len(n)
    max1 = max2 = max3 = -(10**18)
    a = []
    l = r = 0
    while res >= 10**4:
        r = l + 10**4
        while n[r] != ' ':
            r -= 1
        s = n[l:r]
        a.append(s)
        l = r
        res -= len(s)
    if res > 0:
        a.append(n[l:len(n)])
    for i in a:
        arr = [int(x) for x in i.split()]
        if max(arr) > max1:
            max2, max3 = max1, max2
            max1 = max(arr)
            arr.remove(max1)
        if max(arr) > max2:
            max3 = max2
            max2 = max(arr)
            arr.remove(max2)
        if max(arr) > max3:
            max3 = max(arr)
            arr.remove(max3)
        del arr
    print(max1+max2+max3)