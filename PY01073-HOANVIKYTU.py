from itertools import permutations

if __name__ == '__main__':
    res = permutations(input())
    for x in res:
        for y in x: print(y, end = '')
        print()