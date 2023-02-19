if __name__ == '__main__':
    s = input()
    c1, c2 = 0, 0
    for x in s:
        if x.islower(): c1 += 1
        elif x.isupper(): c2 += 1
    if c1 >= c2: print(s.lower())
    else: print(s.upper())