from math import *
import io, os, sys, time
import array as arr

def merge(arr, temp_arr, left, mid, right):
    i = left
    j = mid + 1
    k = left
    cnt = 0
    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            k += 1
            i += 1
        else:
            temp_arr[k] = arr[j]
            cnt += (mid - i + 1)
            k += 1
            j += 1
    while i <= mid:
        temp_arr[k] = arr[i]
        k += 1
        i += 1
    while j <= right:
        temp_arr[k] = arr[j]
        k += 1
        j += 1
    for i in range(left, right + 1): arr[i] = temp_arr[i]
    return cnt

def countInversion(arr, temp_arr, left, right):
    cnt = 0
    if left < right:
        mid = (left + right) // 2
        cnt += countInversion(arr, temp_arr, left, mid)
        cnt += countInversion(arr, temp_arr, mid + 1, right)
        cnt += merge(arr, temp_arr, left, mid, right)
    return cnt

def mergeSort(arr, n):
    temp_arr = [0] * n
    return countInversion(arr, temp_arr, 0, n - 1)

if __name__ == '__main__':
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    sys.stdout.write(str(mergeSort(a, n)))