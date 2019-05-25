

def permutations(nums:list):
	# Given a collection of distinct integers, 
	# return all possible permutations.
	res = []
	def helper(temp:list, first:int, n:int):
		if first == n:
			res.append(temp[:])
		for i in range(first, n):
			temp[i], temp[first] = temp[first], temp[i] # 交换两数的位置
			helper(temp, first + 1, n)
			temp[i], temp[first] = temp[first], temp[i] # 交换回两数的位置

	helper(nums[:], 0, len(nums))
	return res

nums = [7, 3, 8]
print(permutations(nums))