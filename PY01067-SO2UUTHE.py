import queue

def check(n):
    cnt = 0
    for i in range(len(n)):
        if n[i] == '2': cnt += 1
    return cnt > (len(n) // 2) and (n[0] != '0')

list = []

def init():
    a = queue.Queue()
    a.put('1')
    a.put('2')
    l = 0
    while (l <= 9):
        x = a.get()
        if check(x): list.append(x)
        a.put(x + '0')
        a.put(x + '1')
        a.put(x + '2')
        l = len(x)

if __name__ == '__main__':
    TC = int(input())
    init()
    for t in range(TC):
        n = int(input())
        for i in range(n): print(list[i], end = ' ')
        print()