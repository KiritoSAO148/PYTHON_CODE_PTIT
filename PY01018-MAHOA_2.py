if __name__ == '__main__':
    P = "ABCDEFGHIJKLMNOPQRSTUVWXYZ_."
    while True:
        n = input()
        if n == "0": break
        k, s = n.split()
        res = ""
        for x in s:
            i = 0
            for y in P:
                if x == y: break
                i += 1
            i = (i + int(k)) % 28
            res = P[i] + res
        print(res)