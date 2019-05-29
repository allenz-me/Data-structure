

def nextExceed(nums:list) -> list:
	"""
	返回的数组的第i个位置的值应当是，对于原数组中的第i个元素，
	至少往右走多少步，才能遇到一个比自己大的元素.
	（如果之后没有比自己大的元素，或者已经是最后一个元素，
	则在返回数组的对应位置放上-1）。
	"""
	res = [-1 for i in range(len(nums))]
	stack = []
	for i in range(len(nums)):
		while stack and nums[stack[-1]] < nums[i]:
			res[stack[-1]] = i - stack[-1]
			stack.pop()
		stack.append(i)
	return res

print(nextExceed([5,3,1,2,4]))