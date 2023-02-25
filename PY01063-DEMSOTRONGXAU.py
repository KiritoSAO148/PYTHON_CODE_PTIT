if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        s = input()
        n = input()
        print(s.count(n, 0, len(s)))
        #print(len(s.split(n)) - 1)