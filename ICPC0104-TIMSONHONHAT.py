from sys import stdin

if __name__ == '__main__':
    for t in range(int(stdin.readline())):
        s = stdin.readline()
        res = ''
        for x in s:
            if x.isalpha(): res += ' '
            else: res += x
        a = map(int, res.split())
        print(min(a))