n = int(input())
a = list(map(int, input().split()))
st = []
st.append(0)
cnt = 0
for i in range(n):
	while len(st) and a[i] >= a[st[-1]] and a[st[i - 2]] >= a[st[i - 1]]: st.pop()
	cnt += i - st[-1]
	st.append(i)
print(cnt)