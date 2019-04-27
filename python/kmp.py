

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
			slow = l[slow]
		else:
			l.append(0)
			fast += 1
	return l

print(prefix_table("ababcaccabbbbbbbbbab"))
# 输出[-1, 0, 0, 1, 2, 0, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1]