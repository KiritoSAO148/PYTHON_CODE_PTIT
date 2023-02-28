import queue

def solve(n):
    s = set()
    q = queue.Queue()
    q.put(n)
    while not q.empty():
        x = q.get()
        if x == 1: return len(s) + 1
        s.add(x)
        if x % 2 == 0: q.put(x // 2)
        else: q.put(x * 3 + 1)
    return -1

if __name__ == '__main__':
    while True:
        n = int(input())
        if n == 0: break
        elif n == 1: print(1)
        else: print(solve(n))