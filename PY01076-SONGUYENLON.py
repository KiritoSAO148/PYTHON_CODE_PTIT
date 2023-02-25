from math import gcd

if __name__ == '__main__':
    TC = int(input())
    for t in range(TC):
        print(gcd(int(input()), int(input())))