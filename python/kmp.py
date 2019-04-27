
def prefix_table(s:str) -> list:
	"""return the prefix table of string s"""
	# print(len(s))
	assert len(s) >= 2
	l = [-1, 0]
	slow, fast = 0, 1
	while fast < len(s) - 1:
		if s[slow] == s[fast]:
			slow += 1
			l.append(slow)
			fast += 1
		elif slow > 0:
			slow = l[slow] # 使用了之前匹配的信息，关键一步
		else:
			l.append(0)
			fast += 1
	return l



def kmp(target:str, pattern:str) -> int:
	"""return the first index if the pattern matches the target else -1"""
	t, p = 0, 0  # points to target and pattern repectively
	next_ = prefix_table(pattern)
	while t < len(target) and p < len(pattern):
		if p == -1 or target[t] == pattern[p]:
			p += 1
			t += 1
		else:
			p = next_[p]
	if p == len(pattern):
		return t - p
	else:
		return -1

if __name__ == '__main__':
	s = "ababcaccabbbbbbbbbab"
	print(prefix_table(s))
	# 输出[-1, 0, 0, 1, 2, 0, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1]
	print(kmp(target=s, pattern="cab")) # 7