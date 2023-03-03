from itertools import combinations

if __name__ == '__main__':
    n, k = map(int, input().split())
    a = sorted(list(set(list(map(int, input().split())))))
    res = list(combinations(a, k))
    for x in res:
        for y in x: print(y, end = ' ')
        print()